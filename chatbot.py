"""Chatbot de consulta del inventario (SOLO LECTURA).

Responde preguntas como "cuantos iPhone 13 quedan?" o "cuanto vale el inventario?".

COMO FUNCIONA
-------------
  1. Si hay una clave GOOGLE_API_KEY, la pregunta se envia a Gemini junto con el
     texto del inventario (el mismo que entrega GET /api/contexto). Gemini
     responde en lenguaje natural usando SOLO esos datos.
  2. Si no hay clave, o Gemini falla (sin internet, cuota gratuita agotada...),
     responde con reglas simples (_reglas). Asi la demostracion no se cae.

El chatbot nunca modifica el stock ni el catalogo: solo lee.

LA CLAVE
--------
  Se busca en la variable de entorno GOOGLE_API_KEY y, si no esta, en un archivo
  .env dentro de esta carpeta (esta en .gitignore: NUNCA se sube a GitHub).
  Solo se envia a Google; no se devuelve al navegador ni se escribe en el log.
  Modelo: gemini-3.1-flash-lite por defecto (cambiar con GEMINI_MODEL).

Usa solo la biblioteca estandar (no necesita instalar nada).
"""
import json
import os
import unicodedata
import urllib.error
import urllib.request

import db

# Endpoint de Gemini compatible con el formato de OpenAI (el mismo de los notebooks).
URL_GEMINI = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
MODELO = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
MAX_TURNOS = 8        # mensajes anteriores que se reenvian como memoria de la charla
MAX_TEXTO = 600       # largo maximo de cada mensaje del historial

INSTRUCCIONES = """Eres el asistente de consulta del inventario de celulares de \
Distribuidora Austral del Estrecho SpA (Punta Arenas).
Reglas:
- Responde en español, claro y breve.
- Escribe en texto plano: sin Markdown, sin asteriscos ni negritas. Para listar usa guiones simples.
- Usa SOLO los datos del inventario de abajo. Si algo no esta ahi, dilo; no inventes cifras ni modelos.
- Escribe los precios en pesos chilenos con punto de miles, por ejemplo $129.990.
- Solo respondes consultas. No puedes cambiar stock, precios ni el catalogo: si te lo piden, \
explica que eso se hace desde el panel.
- Si preguntan cuando se acaba o agota un modelo, que modelos hay que reponer o por la \
prediccion de stock, usa los datos de pronostico de quiebre de stock de abajo y aclara \
brevemente si se basa en historial de ventas simulado.
- Si preguntan algo que no tiene relacion con el inventario, di amablemente que solo puedes \
responder sobre el inventario.

INVENTARIO ACTUAL
"""


# ------------------------------------------------------------------ clave de API
def _clave():
    """Devuelve la clave de Gemini o '' si no hay (variable de entorno o .env local)."""
    clave = os.environ.get("GOOGLE_API_KEY", "").strip()
    if clave:
        return clave
    try:
        with open(os.path.join(db.BASE, ".env"), encoding="utf-8-sig") as f:
            for linea in f:
                linea = linea.strip()
                if linea.startswith("GOOGLE_API_KEY="):
                    return linea.split("=", 1)[1].strip().strip("'\"")
    except OSError:
        pass
    return ""


def estado():
    """Para /api/health: con que modo responde el chatbot (nunca incluye la clave)."""
    hay = bool(_clave())
    return {"modo": "gemini" if hay else "reglas", "modelo": MODELO if hay else None}


# ------------------------------------------------------------------ Gemini
def _mensajes(mensaje, historial, contexto):
    msgs = [{"role": "system", "content": INSTRUCCIONES + contexto}]
    if isinstance(historial, list):
        for h in historial[-MAX_TURNOS:]:
            if not isinstance(h, dict) or not isinstance(h.get("texto"), str):
                continue
            rol = "assistant" if h.get("rol") == "bot" else "user"
            msgs.append({"role": rol, "content": h["texto"][:MAX_TEXTO]})
    msgs.append({"role": "user", "content": mensaje})
    return msgs


def _gemini(clave, mensajes):
    cuerpo = json.dumps({"model": MODELO, "messages": mensajes, "temperature": 0.2,
                         "max_tokens": 1024}).encode("utf-8")
    req = urllib.request.Request(URL_GEMINI, data=cuerpo, method="POST", headers={
        "Content-Type": "application/json", "Authorization": "Bearer " + clave})
    with urllib.request.urlopen(req, timeout=25) as r:
        datos = json.load(r)
    texto = (datos["choices"][0]["message"]["content"] or "").strip()
    # El chat del panel muestra texto plano: se quitan los asteriscos de Markdown si se cuelan.
    return texto.replace("**", "").replace("__", "")


# ------------------------------------------------------------------ respaldo sin IA
def _norm(t):
    return "".join(c for c in unicodedata.normalize("NFKD", t.lower()) if not unicodedata.combining(c))


def _clp(n):
    return "$" + f"{n:,}".replace(",", ".")


def _resumen_pred_item(p, nombre, incluir_nota=True):
    """Formatea en texto plano la prediccion de quiebre de stock de un modelo."""
    nota = " (pronostico ML con historial base simulado)" if (incluir_nota and p.get("datos_simulados")) else ""
    est = p.get("estado")
    if est == "sin_datos":
        return (f"Todavia no hay suficientes datos de ventas para predecir cuando se acaba el {nombre} "
                f"(stock actual: {p['stock_actual']} unidades).")
    if est == "agotado":
        sug = p.get("cantidad_sugerida") or 0
        extra = f" Se sugiere reponer {sug} unidades para cubrir 30 dias." if sug else ""
        return (f"El {nombre} esta agotado (0 unidades; venta estimada {p['venta_diaria']} u/dia)."
                f"{extra}{nota}")
    if p.get("dias_hasta_agotarse") is None:
        return (f"El {nombre} (stock: {p['stock_actual']} unidades, venta estimada {p['venta_diaria']} u/dia) "
                f"no se agotaria en los proximos 120 dias (estado: {est}).{nota}")
    sug = p.get("cantidad_sugerida") or 0
    extra = f", sugerido reponer {sug} unidades (30 dias)" if sug else ""
    return (f"El {nombre} (stock: {p['stock_actual']} unidades) se agotaria en {p['dias_hasta_agotarse']} "
            f"dias (aprox. {p['fecha_agotamiento']}), con una venta estimada de {p['venta_diaria']} u/dia. "
            f"Estado: {est}, punto de reorden: {p['punto_reorden']} unidades{extra}.{nota}")


def _reglas(pregunta, inv):
    """Respuestas simples por palabras clave. Se usa cuando Gemini no esta disponible."""
    t = _norm(pregunta)
    items = inv["items"]
    if not items:
        return "El catalogo esta vacio: todavia no hay celulares registrados."

    def alguna(*palabras):
        return any(p in t for p in palabras)

    quiere_pred = alguna(
        "acaba", "acabe", "agota", "agote", "agotar", "quiebre",
        "prediccion", "pronostico", "reponer", "reposicion", "reorden",
        "cuando", "durar", "dura", "dias", "alcanza",
    )

    def _pred_por_modelo():
        try:
            import prediccion
            pred = prediccion.calcular_prediccion()
            return pred, {it["modelo"]: it for it in pred["items"]}
        except Exception:
            return None, {}

    # 1) Preguntas sobre un modelo o marca concreta
    mencionados = [i for i in items
                   if any(len(p) >= 3 and p in t for p in _norm(f"{i['nombre']} {i['marca']}").split())]
    if mencionados:
        _, mapa_pred = _pred_por_modelo() if quiere_pred else (None, {})
        lineas = []
        for i in mencionados:
            nombre = f"{i['marca']} {i['nombre']}"
            quiere_precio = alguna("precio", "cuesta", "vale", "valor", "costo")
            quiere_stock = alguna("stock", "quedan", "queda", "hay", "unidades", "cantidad", "disponible")
            if quiere_precio:
                lineas.append(f"El {nombre} cuesta {_clp(i['precio'])}.")
            if quiere_stock:
                lineas.append(f"Del {nombre} quedan {i['cantidad']} unidades "
                              f"({_clp(i['precio_x_cantidad'])} en bodega).")
            if quiere_pred and i["nombre"] in mapa_pred:
                lineas.append(_resumen_pred_item(mapa_pred[i["nombre"]], nombre))
            if not (quiere_precio or quiere_stock or (quiere_pred and i["nombre"] in mapa_pred)):
                lineas.append(f"{nombre}: {_clp(i['precio'])}, {i['cantidad']} unidades en stock. "
                              f"Pantalla {i['pantalla']}, RAM {i['ram']}, almacenamiento "
                              f"{i['almacenamiento']}, bateria {i['bateria']}, camara {i['camara']}.")
        return "\n".join(lineas)

    # 2) Preguntas generales
    if quiere_pred:
        pred, _ = _pred_por_modelo()
        if pred and pred["items"]:
            encabezado = (
                "Pronostico de quiebre de stock (historial base de 90 dias simulado):"
                if pred.get("datos_simulados")
                else "Pronostico de quiebre de stock:"
            )
            lineas = [encabezado]
            for p in pred["items"]:
                lineas.append("- " + _resumen_pred_item(p, f"{p['marca']} {p['modelo']}", incluir_nota=False))
            return "\n".join(lineas)
    if alguna("revisar"):
        return f"Hay {inv['por_revisar']} equipos por revisar."
    if alguna("mas stock", "mayor stock"):
        i = max(items, key=lambda x: x["cantidad"])
        return f"El modelo con mas stock es el {i['marca']} {i['nombre']}: {i['cantidad']} unidades."
    if alguna("menos stock", "menor stock", "poco stock"):
        i = min(items, key=lambda x: x["cantidad"])
        return f"El modelo con menos stock es el {i['marca']} {i['nombre']}: {i['cantidad']} unidades."
    if alguna("mas caro", "mas costoso"):
        i = max(items, key=lambda x: x["precio"])
        return f"El mas caro es el {i['marca']} {i['nombre']}: {_clp(i['precio'])}."
    if alguna("mas barato", "mas economico"):
        i = min(items, key=lambda x: x["precio"])
        return f"El mas barato es el {i['marca']} {i['nombre']}: {_clp(i['precio'])}."
    if alguna("total", "valorizado", "inventario", "bodega"):
        return (f"El inventario vale {_clp(inv['total_inventario'])} "
                f"({inv['total_unidades']} unidades en {len(items)} modelos).")
    if alguna("modelo", "catalogo", "lista", "venden", "tienen", "tienes", "productos", "celulares"):
        return "Tenemos:\n" + "\n".join(
            f"- {i['marca']} {i['nombre']}: {_clp(i['precio'])}, {i['cantidad']} unidades" for i in items)
    return ("No entendi la pregunta. Puedo decirte el precio y el stock de un modelo "
            "(" + ", ".join(i["nombre"] for i in items) + "), el total del inventario "
            "o el listado de modelos.")


# ------------------------------------------------------------------ punto de entrada
def responder(mensaje, historial, contexto, inv):
    """Devuelve {"respuesta", "modo": "gemini"|"reglas", "aviso"?}. No lanza errores de red."""
    clave = _clave()
    aviso = None
    if clave:
        try:
            texto = _gemini(clave, _mensajes(mensaje, historial, contexto))
            if texto:
                return {"respuesta": texto, "modo": "gemini", "modelo": MODELO}
            aviso = "Gemini devolvio una respuesta vacia; respondo en modo basico."
        except urllib.error.HTTPError as e:
            detalle = e.read().decode("utf-8", "replace")[:300].replace(clave, "***")
            print(f"[chatbot] Gemini respondio HTTP {e.code}: {detalle}")
            aviso = {
                429: "Se agoto la cuota gratuita de Gemini por ahora; respondo en modo basico.",
                401: "Gemini rechazo la clave de API; respondo en modo basico.",
                403: "Gemini rechazo la clave de API; respondo en modo basico.",
                400: "Gemini no acepto la solicitud (revisa la clave y el modelo); respondo en modo basico.",
                404: "Gemini no encontro el modelo configurado; respondo en modo basico.",
            }.get(e.code, f"Gemini respondio con error {e.code}; respondo en modo basico.")
        except (OSError, ValueError, KeyError, IndexError, TypeError) as e:
            print(f"[chatbot] No se pudo usar Gemini: {type(e).__name__}")
            aviso = "No se pudo conectar con Gemini; respondo en modo basico."
    else:
        aviso = "Sin clave GOOGLE_API_KEY: respondo en modo basico (sin IA)."
    return {"respuesta": _reglas(mensaje, inv), "modo": "reglas", "aviso": aviso}
