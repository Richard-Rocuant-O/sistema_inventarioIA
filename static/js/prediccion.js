(() => {
  "use strict";

  const API_BASE = "/api";
  const q = (sel) => document.querySelector(sel);

  const TIPOS_MOV = {
    venta: { texto: "Venta", clase: "chip aviso" },
    ingreso_ocr: { texto: "Ingreso OCR", clase: "chip ok" },
    alta: { texto: "Alta", clase: "chip ok" },
    baja: { texto: "Baja", clase: "chip mal" },
    stock: { texto: "Ajuste stock", clase: "chip" },
    edicion: { texto: "Edición", clase: "chip" },
    conteo_foto: { texto: "Conteo foto", clase: "chip" },
  };

  const ESTADOS_PRED = {
    agotado: { texto: "Agotado", clase: "chip mal" },
    critico: { texto: "Crítico (≤ 7 d)", clase: "chip mal" },
    reponer: { texto: "Reponer", clase: "chip aviso" },
    ok: { texto: "OK", clase: "chip ok" },
    sin_datos: { texto: "Sin datos", clase: "chip" },
  };

  function nodo(tag, clase, texto) {
    const el = document.createElement(tag);
    if (clase) el.className = clase;
    if (texto !== undefined && texto !== null) el.textContent = String(texto);
    return el;
  }

  function avisar(mensaje, esError = false) {
    const estado = q("#venta-estado");
    if (estado) estado.textContent = mensaje;
    if (typeof window.toast === "function") window.toast(mensaje, esError);
  }

  async function pedirJson(ruta, opciones = {}) {
    const resp = await fetch(API_BASE + ruta, opciones);
    const datos = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(datos.error || `Error ${resp.status}`);
    return datos;
  }

  function formatearFecha(iso) {
    if (!iso) return "—";
    return String(iso).replace("T", " ");
  }

  // ------------------------------------------------------------ select de modelos
  async function cargarModelosVenta() {
    const sel = q("#venta-modelo");
    if (!sel) return;
    const seleccionado = sel.value;
    try {
      const lista = await pedirJson("/celulares");
      sel.replaceChildren();
      if (!lista.length) {
        sel.append(nodo("option", "", "No hay modelos en el catálogo"));
        sel.options[0].value = "";
        return;
      }
      for (const c of lista) {
        const opt = document.createElement("option");
        opt.value = String(c.id);
        opt.textContent = `${c.marca} ${c.nombre} (stock: ${c.cantidad})`;
        sel.append(opt);
      }
      if (seleccionado && lista.some((c) => String(c.id) === seleccionado)) {
        sel.value = seleccionado;
      }
    } catch (e) {
      sel.replaceChildren();
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "Error al cargar modelos";
      sel.append(opt);
    }
  }

  // ------------------------------------------------------------ historial (ultimas 20)
  async function cargarHistorial() {
    const cuerpo = q("#tabla-movimientos tbody");
    const vacio = q("#movs-vacio");
    if (!cuerpo) return;
    try {
      const lista = await pedirJson("/movimientos?limite=20");
      cuerpo.replaceChildren();
      for (const m of lista) {
        const meta = TIPOS_MOV[m.tipo] || { texto: m.tipo || "Movimiento", clase: "chip" };
        const tr = document.createElement("tr");

        const tdFecha = nodo("td", "", formatearFecha(m.fecha));
        const tdTipo = document.createElement("td");
        tdTipo.append(nodo("span", meta.clase, meta.texto));

        const tdModelo = nodo("td", "", m.nombre || "—");
        const tdAnt = nodo("td", "num", m.anterior != null ? m.anterior : "—");
        const tdNuevo = nodo("td", "num", m.nuevo != null ? m.nuevo : "—");
        const tdDet = nodo("td", "", m.detalle || "—");

        tr.append(tdFecha, tdTipo, tdModelo, tdAnt, tdNuevo, tdDet);
        cuerpo.append(tr);
      }
      if (vacio) vacio.hidden = lista.length > 0;
    } catch (e) {
      avisar("No se pudo cargar el historial: " + e.message, true);
    }
  }

  // ------------------------------------------------------------ prediccion ML
  async function generarPrediccion() {
    const btn = q("#btn-prediccion");
    const cont = q("#res-prediccion");
    const cuerpo = q("#tabla-prediccion tbody");
    const notaMae = q("#pred-nota-mae");
    const chipSim = q("#pred-simulacion");
    if (!btn || !cont || !cuerpo) return;

    btn.disabled = true;
    btn.textContent = "Calculando predicción…";
    try {
      const datos = await pedirJson("/prediccion");
      cont.hidden = false;

      if (chipSim) {
        chipSim.hidden = !datos.datos_simulados;
        chipSim.textContent = datos.datos_simulados
          ? "Historial base simulado (90 días)"
          : "";
      }

      if (notaMae) {
        const ev = datos.evaluacion || {};
        let textoNota = "";
        if (ev.modelos_evaluados > 0) {
          textoNota =
            `Evaluación (últimos ${datos.dias_evaluacion} días): ` +
            `MAE regresión lineal = ${Number(ev.mae_modelo).toFixed(3)} u/día ` +
            `vs promedio simple (${datos.dias_baseline} días previos) = ${Number(ev.mae_promedio).toFixed(3)} u/día.`;
        } else {
          textoNota = "Aún no hay suficientes días de historial para calcular la evaluación comparativa.";
        }
        if (datos.nota_simulacion) {
          textoNota += " " + datos.nota_simulacion;
        }
        notaMae.textContent = textoNota;
      }

      cuerpo.replaceChildren();
      for (const it of datos.items || []) {
        const est = ESTADOS_PRED[it.estado] || { texto: it.estado || "—", clase: "chip" };
        let textoDias = "—";
        if (it.estado === "sin_datos") textoDias = "Sin datos";
        else if (it.estado === "agotado") textoDias = "0 días";
        else if (it.dias_hasta_agotarse == null) textoDias = `> ${datos.horizonte_dias} días`;
        else textoDias = `${it.dias_hasta_agotarse} día(s)`;

        const tr = document.createElement("tr");
        const tdModelo = nodo("td", "modelo", `${it.marca} ${it.modelo}`);
        const tdStock = nodo("td", "num", it.stock_actual);
        const tdVenta = nodo("td", "num", it.venta_diaria != null ? Number(it.venta_diaria).toFixed(2) : "—");
        const tdDias = nodo("td", "num", textoDias);
        const tdFecha = nodo("td", "", it.fecha_agotamiento || "—");
        const tdReorden = nodo("td", "num", it.punto_reorden != null ? it.punto_reorden : "—");
        const tdSug = nodo("td", "num", it.cantidad_sugerida != null ? it.cantidad_sugerida : "—");

        const tdEstado = document.createElement("td");
        tdEstado.append(nodo("span", est.clase, est.texto));

        tr.append(tdModelo, tdStock, tdVenta, tdDias, tdFecha, tdReorden, tdSug, tdEstado);
        cuerpo.append(tr);
      }
    } catch (e) {
      avisar("No se pudo generar la predicción: " + e.message, true);
    } finally {
      btn.disabled = false;
      btn.textContent = "Generar predicción de quiebre de stock (ML)";
    }
  }

  // ------------------------------------------------------------ venta
  async function enviarVenta(ev) {
    ev.preventDefault();
    const sel = q("#venta-modelo");
    const inp = q("#venta-cantidad");
    const btn = q("#btn-vender");
    if (!sel || !inp || !btn) return;

    const celularId = Number(sel.value);
    const cantidad = Number(inp.value);
    if (!Number.isInteger(celularId) || celularId <= 0) {
      avisar("Selecciona un modelo válido.", true);
      return;
    }
    if (!Number.isInteger(cantidad) || cantidad <= 0) {
      avisar("La cantidad debe ser un entero mayor que 0.", true);
      return;
    }

    btn.disabled = true;
    try {
      const res = await pedirJson("/ventas", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ celular_id: celularId, cantidad }),
      });
      inp.value = "1";
      avisar(`Venta registrada: ${res.cantidad} × ${res.modelo} (stock restante: ${res.stock_nuevo}).`);
      if (typeof window.cargar === "function") await window.cargar();
      await Promise.all([cargarModelosVenta(), cargarHistorial()]);
      const resPred = q("#res-prediccion");
      if (resPred && !resPred.hidden) await generarPrediccion();
    } catch (e) {
      avisar(e.message, true);
    } finally {
      btn.disabled = false;
    }
  }

  // ------------------------------------------------------------ arranque
  function iniciar() {
    const formVenta = q("#form-venta");
    const btnPred = q("#btn-prediccion");
    const btnRefrescar = q("#btn-refrescar-movs");
    const selModelo = q("#venta-modelo");

    if (formVenta) formVenta.addEventListener("submit", enviarVenta);
    if (btnPred) btnPred.addEventListener("click", generarPrediccion);
    if (btnRefrescar) btnRefrescar.addEventListener("click", () => {
      cargarHistorial();
      cargarModelosVenta();
    });
    if (selModelo) selModelo.addEventListener("focus", cargarModelosVenta);

    cargarModelosVenta();
    cargarHistorial();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", iniciar);
  } else {
    iniciar();
  }
})();
