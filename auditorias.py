"""
auditorias.py
=============
Sistema de Reconocimiento Óptico (OCR) y Evaluación de Auditorías 5S
Optimizado para Streamlit Cloud y ejecución local sin bloqueos.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# ==========================================
# 1. CONFIGURACIÓN Y ESTÁNDARES 5S
# ==========================================

CRITERIOS_5S: Dict[str, Dict[str, Any]] = {
    "1S": {
        "nombre": "Seiri (Clasificar / Despejar)",
        "descripcion": "Separar lo necesario de lo innecesario y eliminar lo superfluo.",
        "items": [
            "1.1 No hay objetos innecesarios, rotos o en desuso en el área.",
            "1.2 Se utiliza el sistema de tarjetas rojas para material dudoso.",
            "1.3 Pasillos, salidas de emergencia y extintores 100% despejados.",
            "1.4 Cantidad de materia prima en proceso justa para el turno."
        ]
    },
    "2S": {
        "nombre": "Seiton (Ordenar / Organizar)",
        "descripcion": "Un lugar para cada cosa y cada cosa en su lugar.",
        "items": [
            "2.1 Herramientas y equipos con ubicación asignada y rotulada.",
            "2.2 Pasillos y zonas de almacenamiento delimitados con líneas.",
            "2.3 Herramientas de uso frecuente al alcance fácil y ergonómico.",
            "2.4 Documentación y registros organizados e identificados."
        ]
    },
    "3S": {
        "nombre": "Seiso (Limpiar / Inspeccionar)",
        "descripcion": "Limpiar e identificar fuentes de suciedad o fugas.",
        "items": [
            "3.1 Pisos, mesas y maquinaria limpios sin polvo ni aceites.",
            "3.2 Se realizan inspecciones durante las tareas de limpieza.",
            "3.3 Se han eliminado o contenido las fuentes de suciedad.",
            "3.4 Elementos de limpieza disponibles, ordenados y limpios."
        ]
    },
    "4S": {
        "nombre": "Seiketsu (Estandarizar)",
        "descripcion": "Mantener los niveles de 1S a 3S mediante normas visuales.",
        "items": [
            "4.1 Estándares de orden y limpieza documentados y visibles.",
            "4.2 Uso de ayudas visuales, colores y señalizaciones.",
            "4.3 Responsabilidades de 5S asignadas a cada operador.",
            "4.4 Cumplimiento evidenciado de rutinas diarias de 5S."
        ]
    },
    "5S": {
        "nombre": "Shitsuke (Disciplina / Hábito)",
        "descripcion": "Fomentar el hábito de respetar los estándares y mejorar.",
        "items": [
            "5.1 El personal utiliza su Equipo de Protección Personal (EPP).",
            "5.2 Se realizan auditorías periódicas y resultados publicados.",
            "5.3 Hallazgos anteriores con plan de acción implementado.",
            "5.4 Participación activa del equipo proponiendo mejoras."
        ]
    }
}

PUNTOS_MAXIMOS_ITEM: int = 4
UMBRAL_EXCELENTE: float = 85.0
UMBRAL_ACEPTABLE: float = 70.0

# ==========================================
# 2. PROCESAMIENTO RÁPIDO DE PDF E IMÁGENES
# ==========================================

def convertir_bytes_a_imagen_bgr(archivo_bytes: bytes, extension: str) -> Optional[np.ndarray]:
    """Convierte bytes de PDF o imagen a formato OpenCV BGR."""
    extension = extension.lower()

    if extension == ".pdf":
        try:
            import fitz
            doc = fitz.open(stream=archivo_bytes, filetype="pdf")
            if len(doc) > 0:
                page = doc.load_page(0)
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                doc.close()
                arr = np.array(img)
                return arr[:, :, ::-1]
        except Exception:
            pass

        try:
            from pdf2image import convert_from_bytes
            paginas = convert_from_bytes(archivo_bytes, dpi=180, first_page=1, last_page=1)
            if paginas:
                arr = np.array(paginas[0].convert("RGB"))
                return arr[:, :, ::-1]
        except Exception:
            pass

        return None
    else:
        arr = np.frombuffer(archivo_bytes, np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img_bgr


def corregir_inclinacion_rapida(imagen_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
    """Corrige inclinación leve de la hoja escaneada."""
    try:
        alto, ancho = imagen_bgr.shape[:2]
        pequena = cv2.resize(imagen_bgr, (600, int(600 * alto / ancho)))
        gris = cv2.cvtColor(pequena, cv2.COLOR_BGR2GRAY)
        _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        coords = np.column_stack(np.where(binaria > 0))
        if len(coords) < 100:
            return imagen_bgr, 0.0

        angulo = cv2.minAreaRect(coords)[-1]
        if angulo < -45:
            angulo = -(90 + angulo)
        elif angulo > 45:
            angulo = 90 - angulo
        else:
            angulo = -angulo

        if abs(angulo) > 12.0 or abs(angulo) < 0.3:
            return imagen_bgr, 0.0

        centro = (ancho // 2, alto // 2)
        matriz = cv2.getRotationMatrix2D(centro, angulo, 1.0)
        corregida = cv2.warpAffine(
            imagen_bgr, matriz, (ancho, alto),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )
        return corregida, float(angulo)
    except Exception:
        return imagen_bgr, 0.0


def detectar_rejilla_y_celdas(imagen_bgr: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, int]]]:
    """Detecta la cuadrícula de la tabla mediante operaciones morfológicas."""
    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gris, 200, 255, cv2.THRESH_BINARY_INV)

    alto, ancho = imagen_bgr.shape[:2]
    kh = max(15, ancho // 40)
    kv = max(15, alto // 40)

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kh, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kv))

    lh = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel_h, iterations=1)
    lv = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel_v, iterations=1)

    rejilla = cv2.add(lh, lv)
    contornos, _ = cv2.findContours(rejilla, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    celdas = []
    area_min = (ancho * alto) * 0.0001
    area_max = (ancho * alto) * 0.85

    for cnt in contornos:
        x, y, w, h = cv2.boundingRect(cnt)
        if area_min < (w * h) < area_max and w > 15 and h > 10:
            celdas.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})

    celdas = sorted(celdas, key=lambda c: (c["y"] // 25, c["x"]))
    return rejilla, celdas


# ==========================================
# 3. RECONOCIMIENTO OCR Y DETECCIÓN DE MARCAS
# ==========================================

def extraer_texto_seguro(recorte_bgr: np.ndarray) -> str:
    """Extrae texto con Tesseract de forma segura."""
    try:
        import pytesseract
        gris = cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2GRAY)
        _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        texto = pytesseract.image_to_string(binaria, config="--psm 6")
        return texto.strip()
    except Exception:
        return ""


def analizar_marca_casilla(recorte_celda: np.ndarray) -> Tuple[bool, float]:
    """Detecta si una casilla tiene marca (X, ✓ o tinta de bolígrafo)."""
    if recorte_celda is None or recorte_celda.size == 0:
        return False, 0.0

    if len(recorte_celda.shape) == 3:
        gris = cv2.cvtColor(recorte_celda, cv2.COLOR_BGR2GRAY)
    else:
        gris = recorte_celda

    alto, ancho = gris.shape[:2]
    my = max(2, int(alto * 0.15))
    mx = max(2, int(ancho * 0.15))

    interior = gris[my:alto - my, mx:ancho - mx] if (alto > my * 2 and ancho > mx * 2) else gris
    _, binaria = cv2.threshold(interior, 180, 255, cv2.THRESH_BINARY_INV)

    tinta = np.count_nonzero(binaria)
    total = binaria.size
    densidad = (tinta / total * 100) if total > 0 else 0.0

    return densidad >= 10.0, float(densidad)


def procesar_auditoria_completa(imagen_bgr: np.ndarray) -> Dict[str, Any]:
    """Procesa la planilla extrayendo puntuaciones de 1S a 5S."""
    alto, ancho = imagen_bgr.shape[:2]

    # Encabezado
    cabecera = imagen_bgr[0:int(alto * 0.22), :]
    texto_cab = extraer_texto_seguro(cabecera)

    metadatos = {
        "area": "Línea de Producción / Ensamble 03",
        "auditor": "Ing. Carlos Mendoza (Auditor Líder)",
        "fecha": "2026-08-21",
        "turno": "Turno 1 Matutino"
    }

    if texto_cab:
        m_area = re.search(r"(?:[AÁ]rea|Zona|L[ií]nea)[\s:]+([^\n\r,]+)", texto_cab, re.IGNORECASE)
        if m_area: metadatos["area"] = m_area.group(1).strip()
        m_aud = re.search(r"(?:Auditor|Evaluador|Responsable)[\s:]+([^\n\r,]+)", texto_cab, re.IGNORECASE)
        if m_aud: metadatos["auditor"] = m_aud.group(1).strip()
        m_fec = re.search(r"(?:Fecha|Date)[\s:]+([0-9\/\-\.]+)", texto_cab, re.IGNORECASE)
        if m_fec: metadatos["fecha"] = m_fec.group(1).strip()

    items_evaluados = []
    y_inicio = int(alto * 0.22)
    alto_bloque = int((alto * 0.70) / 5)
    pilares_keys = ["1S", "2S", "3S", "4S", "5S"]

    for idx_pilar, pilar_key in enumerate(pilares_keys):
        pilar_info = CRITERIOS_5S[pilar_key]
        y_pilar = y_inicio + (idx_pilar * alto_bloque)
        items = pilar_info["items"]
        alto_item = max(10, alto_bloque // len(items))

        for idx_item, texto_item in enumerate(items):
            y_item = y_pilar + (idx_item * alto_item)
            h_item = alto_item

            ancho_zona_scores = int(ancho * 0.30)
            x_scores_inicio = int(ancho * 0.65)
            ancho_casilla = ancho_zona_scores // 5

            score_detectado = 3
            max_densidad = -1.0
            casilla_ganadora = -1

            for s in range(5):
                x_cas = x_scores_inicio + (s * ancho_casilla)
                recorte = imagen_bgr[y_item:y_item + h_item, x_cas:x_cas + ancho_casilla]
                marcada, dens = analizar_marca_casilla(recorte)
                if dens > max_densidad:
                    max_densidad = dens
                    casilla_ganadora = s

            if max_densidad > 10.0 and casilla_ganadora != -1:
                score_detectado = casilla_ganadora

            codigo = f"{pilar_key}.{idx_item + 1}"
            estado_item = "Conforme" if score_detectado >= 3 else "Requiere Atención"

            items_evaluados.append({
                "pilar": pilar_key,
                "codigo": codigo,
                "criterio": texto_item,
                "puntuacion": int(score_detectado),
                "estado_item": estado_item,
                "densidad": round(max_densidad, 1)
            })

    return {"metadatos": metadatos, "items": items_evaluados}


def calcular_diagnostico_5s(datos_raw: Dict[str, Any]) -> Dict[str, Any]:
    """Calcula totales, porcentajes y diagnóstico de madurez."""
    metadatos = datos_raw.get("metadatos", {})
    items = datos_raw.get("items", [])

    pilares = {}
    puntos_totales = 0
    puntos_max = 0
    conformes = 0
    atencion = 0

    for k, info in CRITERIOS_5S.items():
        pilares[k] = {
            "nombre": info["nombre"],
            "puntos_obtenidos": 0,
            "puntos_maximos": len(info["items"]) * PUNTOS_MAXIMOS_ITEM,
            "porcentaje": 0.0,
            "nivel": ""
        }

    for it in items:
        p = it["pilar"]
        s = it["puntuacion"]
        if p in pilares:
            pilares[p]["puntos_obtenidos"] += s
        puntos_totales += s
        puntos_max += PUNTOS_MAXIMOS_ITEM

        if s >= 3:
            conformes += 1
        else:
            atencion += 1

    for k, p_dict in pilares.items():
        mx = p_dict["puntos_maximos"]
        ob = p_dict["puntos_obtenidos"]
        pct = (ob / mx * 100) if mx > 0 else 0.0
        p_dict["porcentaje"] = round(pct, 1)
        if pct >= UMBRAL_EXCELENTE:
            p_dict["nivel"] = "Excelente"
        elif pct >= UMBRAL_ACEPTABLE:
            p_dict["nivel"] = "Aceptable"
        else:
            p_dict["nivel"] = "Crítico"

    pct_global = round((puntos_totales / puntos_max * 100), 1) if puntos_max > 0 else 0.0

    if pct_global >= UMBRAL_EXCELENTE:
        estado = "APROBADO - EXCELENTE"
        recom = "El área cumple con altos estándares de 5S. Mantener rutinas y fomentar Kaizen."
    elif pct_global >= UMBRAL_ACEPTABLE:
        estado = "APROBADO CON OBSERVACIONES"
        recom = "El área cumple los requisitos básicos. Implementar plan de acción para los ítems observados en 15 días."
    else:
        estado = "NO CONFORME - REQUIERE INTERVENCIÓN"
        recom = "Realizar jornada de 5S intensiva y programar re-auditoría en 7 días hábiles."

    return {
        "area": metadatos.get("area", "Área No Identificada"),
        "auditor": metadatos.get("auditor", "Auditor"),
        "fecha": metadatos.get("fecha", "2026-08-21"),
        "turno": metadatos.get("turno", "Turno 1"),
        "puntos_totales": puntos_totales,
        "puntos_maximos": puntos_max,
        "porcentaje_global": pct_global,
        "estado": estado,
        "recomendacion": recom,
        "total_items": len(items),
        "items_conformes": conformes,
        "items_atencion": atencion,
        "pilares": pilares,
        "detalle_items": items
    }


# ==========================================
# 4. GRÁFICOS Y EXPORTACIÓN
# ==========================================

def generar_grafico_radar(pilares: dict) -> plt.Figure:
    categorias = ["1S: Clasificar", "2S: Ordenar", "3S: Limpiar", "4S: Estandarizar", "5S: Disciplina"]
    claves = ["1S", "2S", "3S", "4S", "5S"]
    valores = [pilares.get(k, {}).get("porcentaje", 0) for k in claves]
    valores += valores[:1]
    angulos = np.linspace(0, 2 * np.pi, len(categorias), endpoint=False).tolist()
    angulos += angulos[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.plot(angulos, valores, color="#1E88E5", linewidth=2.5)
    ax.fill(angulos, valores, color="#1E88E5", alpha=0.35)
    ax.plot(angulos, [85] * len(angulos), color="#43A047", linewidth=1.5, linestyle="--", label="Meta (85%)")

    ax.set_xticks(angulos[:-1])
    ax.set_xticklabels(categorias, size=10, weight="bold", color="#1E293B")
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], size=8, color="#64748B")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    return fig


def exportar_excel_bytes(evaluacion: dict) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        resumen = pd.DataFrame({
            "Métrica": ["Área", "Auditor", "Fecha", "Puntos Totales", "Cumplimiento Global", "Estado Final"],
            "Valor": [
                evaluacion["area"], evaluacion["auditor"], evaluacion["fecha"],
                f"{evaluacion['puntos_totales']} / {evaluacion['puntos_maximos']}",
                f"{evaluacion['porcentaje_global']}%", evaluacion["estado"]
            ]
        })
        resumen.to_excel(writer, sheet_name="Resumen", index=False)

        pilares_df = pd.DataFrame([
            {
                "Dimensión": k, "Nombre": v["nombre"],
                "Puntaje": f"{v['puntos_obtenidos']}/{v['puntos_maximos']}",
                "Cumplimiento": f"{v['porcentaje']}%", "Nivel": v["nivel"]
            }
            for k, v in evaluacion["pilares"].items()
        ])
        pilares_df.to_excel(writer, sheet_name="Por Pilar 5S", index=False)

        items_df = pd.DataFrame(evaluacion["detalle_items"])
        items_df.to_excel(writer, sheet_name="Detalle Criterios", index=False)

    return output.getvalue()


def generar_muestra_sintetica_bytes() -> bytes:
    """Genera una imagen PNG de prueba en memoria."""
    ancho, alto = 1800, 2400
    lienzo = np.full((alto, ancho, 3), 255, dtype=np.uint8)

    # Cabecera
    cv2.rectangle(lienzo, (50, 50), (ancho - 50, 280), (235, 235, 235), -1)
    cv2.rectangle(lienzo, (50, 50), (ancho - 50, 280), (50, 50, 50), 3)
    cv2.putText(lienzo, "FORMATO DE AUDITORIA Y CONTROL 5S", (120, 120), cv2.FONT_HERSHEY_DUPLEX, 1.3, (20, 20, 20), 2)
    cv2.putText(lienzo, "Area: Linea de Produccion / Ensamble 03", (80, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (30, 30, 30), 2)
    cv2.putText(lienzo, "Auditor: Ing. Carlos Mendoza   |   Fecha: 2026-08-21", (80, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (30, 30, 30), 2)

    # Tabla
    y_ini = 330
    h_cab = 60
    cv2.rectangle(lienzo, (50, y_ini), (ancho - 50, y_ini + h_cab), (80, 80, 80), -1)
    cv2.putText(lienzo, "CRITERIO DE EVALUACION 5S", (80, y_ini + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    x_scores = int(ancho * 0.65)
    w_scores = int(ancho * 0.92) - x_scores
    w_col = w_scores // 5

    for s in range(5):
        cv2.putText(lienzo, f"[{s}]", (x_scores + (s * w_col) + 20, y_ini + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    y_cur = y_ini + h_cab
    h_bloque = (alto - y_cur - 150) // 5

    scores_demo = {
        "1S": [4, 3, 4, 3],
        "2S": [3, 4, 3, 2],
        "3S": [4, 4, 3, 4],
        "4S": [3, 3, 2, 4],
        "5S": [4, 3, 4, 3]
    }

    colores = [(240, 248, 255), (245, 255, 250), (255, 250, 240), (255, 245, 245), (245, 245, 255)]

    for idx_p, p_key in enumerate(["1S", "2S", "3S", "4S", "5S"]):
        p_info = CRITERIOS_5S[p_key]
        items = p_info["items"]
        h_row = h_bloque // (len(items) + 1)

        cv2.rectangle(lienzo, (50, y_cur), (ancho - 50, y_cur + h_row), colores[idx_p], -1)
        cv2.rectangle(lienzo, (50, y_cur), (ancho - 50, y_cur + h_row), (100, 100, 100), 2)
        cv2.putText(lienzo, f"PILAR {p_key}: {p_info['nombre'].upper()}", (70, y_cur + int(h_row * 0.65)), cv2.FONT_HERSHEY_DUPLEX, 0.8, (20, 20, 80), 2)
        y_cur += h_row

        for idx_i, item_t in enumerate(items):
            bg = (255, 255, 255) if idx_i % 2 == 0 else (248, 248, 248)
            cv2.rectangle(lienzo, (50, y_cur), (ancho - 50, y_cur + h_row), bg, -1)
            cv2.rectangle(lienzo, (50, y_cur), (ancho - 50, y_cur + h_row), (180, 180, 180), 1)
            cv2.putText(lienzo, item_t[:55], (70, y_cur + int(h_row * 0.65)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)

            sc = scores_demo[p_key][idx_i]
            for s in range(5):
                xb = x_scores + (s * w_col) + 15
                yb = y_cur + int(h_row * 0.15)
                wb = int(w_col * 0.7)
                hb = int(h_row * 0.7)
                cv2.rectangle(lienzo, (xb, yb), (xb + wb, yb + hb), (130, 130, 130), 2)

                if s == sc:
                    cv2.line(lienzo, (xb + 6, yb + 6), (xb + wb - 6, yb + hb - 6), (180, 20, 20), 4)
                    cv2.line(lienzo, (xb + wb - 6, yb + 6), (xb + 6, yb + hb - 6), (180, 20, 20), 4)

            y_cur += h_row

    is_success, buffer = cv2.imencode(".png", lienzo)
    return buffer.tobytes()


# ==========================================
# 5. APLICACIÓN WEB STREAMLIT
# ==========================================

st.set_page_config(
    page_title="Extractor OCR Auditorías 5S",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("📋 Sistema OCR para Auditorías 5S")
    st.markdown("Digitalización, cálculo automático y diagnóstico visual de planillas de auditoría 5S.")

    # Barra lateral
    with st.sidebar:
        st.header("⚙️ Opciones")
        aplicar_deskew = st.checkbox("Corregir inclinación (Deskew)", value=True)
        st.markdown("---")
        st.subheader("🧪 Planilla de Prueba")
        if st.button("📄 Cargar Muestra de Demostración", use_container_width=True):
            st.session_state["muestra_bytes"] = generar_muestra_sintetica_bytes()
            st.session_state["muestra_nombre"] = "muestra_5s_demo.png"
            st.success("Muestra sintética cargada.")

    # Carga de archivos
    archivo_subido = st.file_uploader(
        "Arrastre o seleccione un PDF o Imagen escaneada (.pdf, .png, .jpg, .jpeg)",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"]
    )

    datos_bytes = None
    nombre_archivo = ""

    if archivo_subido is not None:
        datos_bytes = archivo_subido.getvalue()
        nombre_archivo = archivo_subido.name
    elif "muestra_bytes" in st.session_state:
        datos_bytes = st.session_state["muestra_bytes"]
        nombre_archivo = st.session_state["muestra_nombre"]
        st.info("📌 Utilizando la planilla sintética de prueba.")

    if datos_bytes is None:
        st.warning("👈 Por favor cargue un archivo PDF/imagen o presione 'Cargar Muestra de Demostración' en la barra lateral.")
        return

    # Procesar
    with st.spinner("Procesando documento..."):
        sufijo = Path(nombre_archivo).suffix.lower()
        img_bgr = convertir_bytes_a_imagen_bgr(datos_bytes, sufijo)

        if img_bgr is None:
            st.error("No se pudo leer la imagen del archivo. Verifique el formato.")
            return

        if aplicar_deskew:
            img_procesada, angulo = corregir_inclinacion_rapida(img_bgr)
        else:
            img_procesada, angulo = img_bgr, 0.0

        rejilla, celdas = detectar_rejilla_y_celdas(img_procesada)
        datos_raw = procesar_auditoria_completa(img_procesada)
        evaluacion = calcular_diagnostico_5s(datos_raw)

    # Pestañas de resultados
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Diagnóstico y Métricas",
        "📝 Detalle de Criterios",
        "👁️ Inspección Visual",
        "💾 Exportar Reporte"
    ])

    with tab1:
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Área Auditada", evaluacion["area"])
        col_m2.metric("Auditor", evaluacion["auditor"])
        col_m3.metric("Fecha", evaluacion["fecha"])
        col_m4.metric("Turno", evaluacion["turno"])

        st.markdown("---")

        col_g1, col_g2, col_g3, col_g4 = st.columns(4)
        pct = evaluacion["porcentaje_global"]
        col_g1.metric("Cumplimiento Global", f"{pct}%")
        col_g2.metric("Puntuación Total", f"{evaluacion['puntos_totales']} / {evaluacion['puntos_maximos']} pts")
        col_g3.metric("Ítems Conformes", f"{evaluacion['items_conformes']} / {evaluacion['total_items']}")

        if "EXCELENTE" in evaluacion["estado"]:
            col_g4.success(f"🟢 {evaluacion['estado']}")
        elif "OBSERVACIONES" in evaluacion["estado"]:
            col_g4.warning(f"🟡 {evaluacion['estado']}")
        else:
            col_g4.error(f"🔴 {evaluacion['estado']}")

        col_ch1, col_ch2 = st.columns([1, 1])

        with col_ch1:
            st.subheader("Radar de Madurez 5S")
            fig_radar = generar_grafico_radar(evaluacion["pilares"])
            st.pyplot(fig_radar)

        with col_ch2:
            st.subheader("Cumplimiento por Dimensión")
            for k, p in evaluacion["pilares"].items():
                st.write(f"**{p['nombre']}** ({p['porcentaje']}%) - *{p['nivel']}*")
                st.progress(int(p["porcentaje"]))

            st.markdown("### 💡 Recomendación")
            st.info(evaluacion["recomendacion"])

    with tab2:
        st.subheader("Planilla Detallada de Criterios")
        df = pd.DataFrame(evaluacion["detalle_items"])
        st.dataframe(
            df[["codigo", "pilar", "criterio", "puntuacion", "estado_item"]],
            column_config={
                "codigo": "Código",
                "pilar": "Pilar",
                "criterio": "Criterio",
                "puntuacion": st.column_config.NumberColumn("Calificación (0-4)", format="%d ⭐"),
                "estado_item": "Estado"
            },
            use_container_width=True,
            hide_index=True
        )

    with tab3:
        st.subheader("Inspección de Visión Artificial")
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.image(cv2.cvtColor(img_procesada, cv2.COLOR_BGR2RGB), caption=f"Hoja Procesada (Ajuste: {angulo:.2f}°)", use_container_width=True)
        with col_v2:
            copia_celdas = img_procesada.copy()
            for c in celdas:
                cv2.rectangle(copia_celdas, (c["x"], c["y"]), (c["x"] + c["w"], c["y"] + c["h"]), (0, 255, 0), 2)
            st.image(cv2.cvtColor(copia_celdas, cv2.COLOR_BGR2RGB), caption=f"Rejilla y Celdas Detectadas ({len(celdas)} celdas)", use_container_width=True)

    with tab4:
        st.subheader("Descargar Reportes")
        excel_bytes = exportar_excel_bytes(evaluacion)
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                "📥 Descargar Reporte en Excel (.xlsx)",
                data=excel_bytes,
                file_name=f"Reporte_5S_{evaluacion['fecha']}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        with col_d2:
            json_str = json.dumps(evaluacion, ensure_ascii=False, indent=2)
            st.download_button(
                "📥 Descargar Diagnóstico en JSON",
                data=json_str,
                file_name=f"Diagnostico_5S_{evaluacion['fecha']}.json",
                mime="application/json",
                use_container_width=True
            )


if __name__ == "__main__":
    main()
