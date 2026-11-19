"""Modulo de ventas y prediccion de quiebre de stock (Machine Learning).

Se registra en app.py como Blueprint mediante prediccion.registrar(app):
  - Crea la tabla 'ventas' (guarda el NOMBRE del modelo para que el historial
    sobreviva si el celular se elimina del catalogo).
  - Si 'ventas' esta vacia, genera 90 dias de ventas SIMULADAS (simulado=1) con
    semilla fija, patron semanal, tendencia leve y distribucion de Poisson.
    Esto es un supuesto del caso academico para demostrar el modelo ML.
  - Expone:
      POST /api/ventas      registra una venta real, descuenta stock y deja movimiento.
      GET  /api/prediccion  entrena una LinearRegression por modelo y estima
                            en cuantos dias se agota el stock.
"""
import math
from contextlib import closing
from datetime import date, timedelta

import numpy as np
from flask import Blueprint, abort, jsonify, request
from sklearn.linear_model import LinearRegression

import db

bp = Blueprint("prediccion", __name__)

# Parametros del modelo y del historial simulado
SEMILLA_SIM = 42
DIAS_SIMULADOS = 90
DIAS_EVALUACION = 20       # ultimos N dias reservados para medir el MAE
DIAS_BASELINE = 14         # ventana del promedio simple previo al bloque de evaluacion
MIN_DIAS_MODELO = 14       # minimo de dias de historial para entrenar
HORIZONTE_DIAS = 120       # dias futuros a pronosticar
DIAS_REPOSICION = 7        # tiempo de reposicion (lead time)
DIAS_COBERTURA = 30        # dias objetivo para la cantidad sugerida
Z_SERVICIO = 1.65          # nivel de servicio ~95 % para el stock de seguridad

# Multiplicador por dia de la semana (0=lunes .. 6=domingo; promedio = 1,0)
PATRON_SEMANAL = np.array([0.55, 0.65, 0.80, 1.00, 1.65, 1.95, 0.40], dtype=float)

# Tasa diaria base y tendencia diaria leve por modelo conocido (supuesto del caso)
PERFILES_SIM = {
    "Galaxy A15": (1.10, 0.0020),
    "iPhone 13": (0.28, 0.0007),
    "Redmi Note 13": (0.55, 0.0012),
}

ESQUEMA_VENTAS = """
CREATE TABLE IF NOT EXISTS ventas (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha    TEXT NOT NULL,
    modelo   TEXT NOT NULL,
    cantidad INTEGER NOT NULL CHECK (cantidad > 0),
    simulado INTEGER NOT NULL DEFAULT 0 CHECK (simulado IN (0, 1))
);
CREATE INDEX IF NOT EXISTS idx_ventas_modelo_fecha ON ventas (modelo, fecha);
"""


# ------------------------------------------------------------------ base de datos
def iniciar_tabla_ventas():
    """Crea la tabla 'ventas' y, si esta vacia, siembra 90 dias simulados."""
    with closing(db.conectar()) as con, con:
        con.executescript(ESQUEMA_VENTAS)
        if con.execute("SELECT COUNT(*) FROM ventas").fetchone()[0] > 0:
            return

        modelos = [r["nombre"] for r in con.execute("SELECT nombre FROM celulares ORDER BY id")]
        if not modelos:
            modelos = [fila[0] for fila in db.SEMILLA]

        rng = np.random.default_rng(SEMILLA_SIM)
        hoy = date.today()
        filas = []
        for nombre in modelos:
            base, tendencia = PERFILES_SIM.get(nombre, (0.50, 0.0010))
            for t in range(DIAS_SIMULADOS):
                f = hoy - timedelta(days=DIAS_SIMULADOS - t)
                lam = max(0.05, (base + tendencia * t) * PATRON_SEMANAL[f.weekday()])
                cant = int(rng.poisson(lam))
                if cant > 0:
                    filas.append((f.isoformat(), nombre, cant, 1))

        if filas:
            con.executemany(
                "INSERT INTO ventas (fecha, modelo, cantidad, simulado) VALUES (?,?,?,?)",
                filas,
            )


def _entero_estricto(valor):
    """Convierte a int rechazando booleanos y decimales (ej. True o 1.5)."""
    if isinstance(valor, bool):
        raise ValueError("booleano no valido")
    if isinstance(valor, float):
        if not valor.is_integer():
            raise ValueError("no es entero")
        return int(valor)
    if isinstance(valor, int):
        return valor
    if isinstance(valor, str) and valor.strip():
        return int(valor.strip())
    raise ValueError("no es entero")


# ------------------------------------------------------------------ ML: regresion lineal
def _matriz_features(indices, fechas):
    """Construye la matriz X: [tendencia (dia), one-hot dia de la semana (7 cols)]."""
    t = np.asarray(indices, dtype=float).reshape(-1, 1)
    dows = np.fromiter((f.weekday() for f in fechas), dtype=int, count=len(fechas))
    onehot = np.eye(7, dtype=float)[dows]
    return np.hstack([t, onehot])


def _serie_diaria(con, modelo, hoy):
    """Devuelve (fechas, cantidades, tiene_simulados) para un modelo."""
    filas = con.execute(
        "SELECT fecha, SUM(cantidad) AS total, MAX(simulado) AS tiene_sim"
        " FROM ventas WHERE modelo = ? GROUP BY fecha ORDER BY fecha",
        (modelo,),
    ).fetchall()
    if not filas:
        return [], np.array([], dtype=float), False

    por_fecha = {}
    tiene_sim = False
    for r in filas:
        f = date.fromisoformat(r["fecha"])
        por_fecha[f] = float(r["total"])
        if r["tiene_sim"]:
            tiene_sim = True

    inicio = min(por_fecha)
    ultima_venta = max(por_fecha)
    # Si hubo ventas reales hoy, se incluye hoy; si no, cerramos el historial en ayer
    # para no contar un dia en curso sin ventas como un cero definitivo.
    fin = max(ultima_venta, hoy - timedelta(days=1))
    if fin < inicio:
        fin = inicio

    n = (fin - inicio).days + 1
    fechas = [inicio + timedelta(days=k) for k in range(n)]
    y = np.array([por_fecha.get(f, 0.0) for f in fechas], dtype=float)
    return fechas, y, tiene_sim


def _predecir_modelo(celular, fechas, y, tiene_sim, hoy):
    """Entrena la regresion lineal para un modelo y calcula indicadores de stock."""
    stock = int(celular["cantidad"])
    base = {
        "celular_id": celular["id"],
        "modelo": celular["nombre"],
        "marca": celular["marca"],
        "stock_actual": stock,
        "venta_diaria": None,
        "dias_hasta_agotarse": None,
        "fecha_agotamiento": None,
        "punto_reorden": None,
        "cantidad_sugerida": None,
        "estado": "sin_datos",
        "sigma_residuos": None,
        "mae_modelo": None,
        "mae_promedio": None,
        "dias_historial": len(fechas),
        "datos_simulados": bool(tiene_sim),
    }

    n = len(fechas)
    if n < MIN_DIAS_MODELO or float(np.sum(y)) <= 0:
        return base

    indices = np.arange(n, dtype=float)
    X = _matriz_features(indices, fechas)

    # 1) Evaluacion: entrenar sin los ultimos 20 dias y comparar MAE vs promedio de 14 dias previos
    mae_modelo = None
    mae_promedio = None
    if n >= DIAS_EVALUACION + DIAS_BASELINE:
        X_tr, y_tr = X[:-DIAS_EVALUACION], y[:-DIAS_EVALUACION]
        X_te, y_te = X[-DIAS_EVALUACION:], y[-DIAS_EVALUACION:]
        reg_eval = LinearRegression().fit(X_tr, y_tr)
        pred_eval = np.clip(reg_eval.predict(X_te), 0.0, None)
        mae_modelo = round(float(np.mean(np.abs(y_te - pred_eval))), 3)

        prom_14 = float(np.mean(y_tr[-DIAS_BASELINE:]))
        mae_promedio = round(float(np.mean(np.abs(y_te - prom_14))), 3)

    # 2) Modelo final con todo el historial
    reg_final = LinearRegression().fit(X, y)
    ajuste = np.clip(reg_final.predict(X), 0.0, None)
    residuos = y - ajuste
    sigma = float(np.std(residuos, ddof=1)) if n > 1 else 0.0

    # 3) Pronostico de 120 dias a partir de manana (valores negativos recortados a 0)
    fechas_fut = [hoy + timedelta(days=k) for k in range(1, HORIZONTE_DIAS + 1)]
    inicio_fut = float((fechas_fut[0] - fechas[0]).days)
    indices_fut = np.arange(inicio_fut, inicio_fut + HORIZONTE_DIAS, dtype=float)
    pronostico = np.clip(reg_final.predict(_matriz_features(indices_fut, fechas_fut)), 0.0, None)

    venta_diaria = float(np.mean(pronostico[:DIAS_COBERTURA]))
    punto_reorden_val = (
        venta_diaria * DIAS_REPOSICION
        + Z_SERVICIO * sigma * math.sqrt(DIAS_REPOSICION)
    )
    punto_reorden = int(math.ceil(punto_reorden_val)) if (venta_diaria > 0 or sigma > 0) else 0

    if stock <= 0:
        dias_agot = 0
        fecha_agot = hoy.isoformat()
    else:
        acum = np.cumsum(pronostico)
        cruces = np.flatnonzero(acum >= stock)
        if cruces.size > 0:
            idx = int(cruces[0])
            dias_agot = idx + 1
            fecha_agot = fechas_fut[idx].isoformat()
        else:
            dias_agot = None
            fecha_agot = None

    if stock <= 0:
        estado = "agotado"
    elif dias_agot is not None and dias_agot <= DIAS_REPOSICION:
        estado = "critico"
    elif stock <= punto_reorden:
        estado = "reponer"
    else:
        estado = "ok"

    demanda_30d = float(np.sum(pronostico[:DIAS_COBERTURA]))
    if estado in ("agotado", "critico", "reponer"):
        cantidad_sugerida = max(0, int(math.ceil(demanda_30d - stock)))
    else:
        cantidad_sugerida = 0

    base.update({
        "venta_diaria": round(venta_diaria, 2),
        "dias_hasta_agotarse": dias_agot,
        "fecha_agotamiento": fecha_agot,
        "punto_reorden": punto_reorden,
        "cantidad_sugerida": cantidad_sugerida,
        "estado": estado,
        "sigma_residuos": round(sigma, 3),
        "mae_modelo": mae_modelo,
        "mae_promedio": mae_promedio,
    })
    return base


def calcular_prediccion():
    """Calcula la prediccion de quiebre de stock para todos los celulares del catalogo."""
    hoy = date.today()
    with closing(db.conectar()) as con, con:
        celulares = [dict(r) for r in con.execute("SELECT id, nombre, marca, cantidad FROM celulares ORDER BY id")]
        items = []
        for cel in celulares:
            fechas, y, tiene_sim = _serie_diaria(con, cel["nombre"], hoy)
            items.append(_predecir_modelo(cel, fechas, y, tiene_sim, hoy))
        hay_simulados = bool(
            con.execute("SELECT EXISTS(SELECT 1 FROM ventas WHERE simulado = 1)").fetchone()[0]
        )

    evaluados = [it for it in items if it["mae_modelo"] is not None and it["mae_promedio"] is not None]
    eval_global = {
        "mae_modelo": round(float(np.mean([it["mae_modelo"] for it in evaluados])), 3) if evaluados else None,
        "mae_promedio": round(float(np.mean([it["mae_promedio"] for it in evaluados])), 3) if evaluados else None,
        "modelos_evaluados": len(evaluados),
    }

    return {
        "fecha_calculo": hoy.isoformat(),
        "modelo_ml": "LinearRegression (tendencia + dia de la semana one-hot)",
        "horizonte_dias": HORIZONTE_DIAS,
        "dias_reposicion": DIAS_REPOSICION,
        "dias_cobertura": DIAS_COBERTURA,
        "dias_evaluacion": DIAS_EVALUACION,
        "dias_baseline": DIAS_BASELINE,
        "datos_simulados": hay_simulados,
        "nota_simulacion": (
            "El historial base de 90 dias corresponde a ventas simuladas (supuesto del caso, "
            "semilla fija). Las ventas registradas desde el panel o POST /api/ventas se guardan "
            "como datos reales (simulado=0)."
            if hay_simulados
            else "Pronostico calculado solo con ventas reales registradas."
        ),
        "evaluacion": eval_global,
        "items": items,
    }


def lineas_contexto():
    """Devuelve lineas en texto plano con el pronostico para armar_contexto() y el chatbot."""
    try:
        pred = calcular_prediccion()
    except Exception:
        return []
    if not pred["items"]:
        return []

    origen = "con historial base de 90 dias simulado" if pred["datos_simulados"] else "con ventas reales"
    lineas = [
        f"Pronostico de quiebre de stock (ML, regresion lineal {origen}; fecha {pred['fecha_calculo']}):"
    ]
    for it in pred["items"]:
        nombre = f"{it['marca']} {it['modelo']}"
        if it["estado"] == "sin_datos":
            lineas.append(f"- {nombre}: stock {it['stock_actual']}, sin datos suficientes para predecir.")
            continue
        if it["estado"] == "agotado":
            quiebre = "agotado hoy (0 dias)"
        elif it["dias_hasta_agotarse"] is None:
            quiebre = f"no se agota en los proximos {pred['horizonte_dias']} dias"
        else:
            quiebre = f"se agota en {it['dias_hasta_agotarse']} dias (aprox. {it['fecha_agotamiento']})"
        sug = (
            f", sugerido reponer {it['cantidad_sugerida']} unidades (30 dias)"
            if it["cantidad_sugerida"]
            else ""
        )
        lineas.append(
            f"- {nombre}: stock {it['stock_actual']}, venta estimada {it['venta_diaria']} u/dia, "
            f"{quiebre}, punto de reorden {it['punto_reorden']} unidades{sug}, estado {it['estado']}."
        )
    ev = pred["evaluacion"]
    if ev["modelos_evaluados"] > 0:
        lineas.append(
            f"Evaluacion del modelo (ultimos {pred['dias_evaluacion']} dias): "
            f"MAE regresion lineal {ev['mae_modelo']} vs promedio simple ({pred['dias_baseline']} dias) "
            f"{ev['mae_promedio']}."
        )
    return lineas


# ------------------------------------------------------------------ endpoints
@bp.post("/api/ventas")
def registrar_venta():
    """Registra una venta: {"celular_id": 2, "cantidad": 1}."""
    datos = request.get_json(silent=True)
    if not isinstance(datos, dict):
        abort(400, "Se esperaba un cuerpo JSON (objeto).")

    if "celular_id" not in datos or "cantidad" not in datos:
        abort(400, "Envia 'celular_id' y 'cantidad' como numeros enteros.")
    try:
        celular_id = _entero_estricto(datos["celular_id"])
        cantidad = _entero_estricto(datos["cantidad"])
    except ValueError:
        abort(400, "'celular_id' y 'cantidad' deben ser numeros enteros.")

    if cantidad <= 0:
        abort(400, "'cantidad' debe ser un entero mayor que 0.")

    hoy = date.today().isoformat()
    with closing(db.conectar()) as con, con:
        celular = con.execute(
            "SELECT id, nombre, marca, precio, cantidad FROM celulares WHERE id = ?",
            (celular_id,),
        ).fetchone()
        if celular is None:
            abort(404, f"No existe el celular con id {celular_id}.")

        stock_anterior = int(celular["cantidad"])
        if cantidad > stock_anterior:
            abort(
                409,
                f"Stock insuficiente para {celular['nombre']}: quedan {stock_anterior} "
                f"unidades y se intentaron vender {cantidad}.",
            )

        stock_nuevo = stock_anterior - cantidad
        con.execute(
            "UPDATE celulares SET cantidad = ?, actualizado = ? WHERE id = ?",
            (stock_nuevo, db.ahora(), celular_id),
        )
        db.registrar_movimiento(
            con,
            celular_id,
            "venta",
            stock_anterior,
            stock_nuevo,
            f"Venta de {cantidad} unidad(es) de {celular['nombre']}",
        )
        cur = con.execute(
            "INSERT INTO ventas (fecha, modelo, cantidad, simulado) VALUES (?,?,?,0)",
            (hoy, celular["nombre"], cantidad),
        )
        venta_id = cur.lastrowid

    return jsonify({
        "id": venta_id,
        "fecha": hoy,
        "celular_id": celular_id,
        "modelo": celular["nombre"],
        "marca": celular["marca"],
        "cantidad": cantidad,
        "precio_unitario": int(celular["precio"]),
        "total": int(celular["precio"]) * cantidad,
        "stock_anterior": stock_anterior,
        "stock_nuevo": stock_nuevo,
        "simulado": False,
    }), 201


@bp.get("/api/prediccion")
def obtener_prediccion():
    """Entrena regresion lineal por modelo y devuelve el pronostico de quiebre de stock."""
    return jsonify(calcular_prediccion())


def registrar(app):
    """Registra el Blueprint en Flask e inicializa la tabla 'ventas'."""
    iniciar_tabla_ventas()
    app.register_blueprint(bp)
