"""
app.py
======
Interfaz Gráfica Web Interactiva con Streamlit para el Sistema OCR de Auditorías 5S.
Permite cargar PDFs/imágenes, inspeccionar el preprocesamiento de visión artificial,
visualizar gráficos radar de madurez 5S y descargar reportes en Excel y JSON.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from evaluacion_5s import calcular_evaluacion_5s
from generador_muestras import generar_hoja_auditoria_sintetica
from image_processing import (
    corregir_inclinacion,
    detectar_estructura_tabla,
    resaltar_celdas_detectadas
)
from ocr_engine import procesar_hoja_auditoria
from pdf_reader import leer_archivo_documento
from utils import exportar_auditoria_excel


def crear_grafico_radar_5s(pilares: dict) -> plt.Figure:
    """Genera un gráfico de radar / telaraña con los puntajes porcentuales de las 5S."""
    categorias = ["1S: Clasificar", "2S: Ordenar", "3S: Limpiar", "4S: Estandarizar", "5S: Disciplina"]
    claves = ["1S", "2S", "3S", "4S", "5S"]
    valores = [pilares.get(k, {}).get("porcentaje", 0) for k in claves]
    
    valores += valores[:1]
    angulos = np.linspace(0, 2 * np.pi, len(categorias), endpoint=False).tolist()
    angulos += angulos[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    ax.plot(angulos, valores, color="#1E88E5", linewidth=2.5, linestyle="solid")
    ax.fill(angulos, valores, color="#1E88E5", alpha=0.35)

    ax.plot(angulos, [85] * len(angulos), color="#43A047", linewidth=1.5, linestyle="--", label="Meta (85%)")

    ax.set_xticks(angulos[:-1])
    ax.set_xticklabels(categorias, size=11, weight="bold", color="#2E3A59")
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], size=9, color="#666666")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    ax.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    return fig


def main():
    st.set_page_config(
        page_title="Sistema OCR Auditorías 5S",
        page_icon="📋",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("📋 Sistema Local de Reconocimiento Óptico (OCR) para Auditorías 5S")
    st.markdown(
        "Herramienta inteligente de visión por computadora y OCR para digitalizar, "
        "evaluar y generar diagnósticos automáticos de planillas de auditoría 5S."
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Parámetros del Sistema")
        dpi = st.slider("Resolución de Escaneo (DPI)", min_value=150, max_value=400, value=300, step=50)
        aplicar_deskew = st.checkbox("Corrección de Inclinación (Deskew)", value=True)
        
        st.markdown("---")
        st.subheader("🧪 Muestra de Prueba")
        if st.button("📄 Generar y Cargar Muestra Sintética", use_container_width=True):
            with st.spinner("Generando planilla 5S de prueba..."):
                generar_hoja_auditoria_sintetica("demo_auditoria_5s.png", "demo_auditoria_5s.pdf")
                st.session_state["archivo_demo"] = "demo_auditoria_5s.pdf"
                st.success("Muestra creada con éxito.")

    # File Uploader
    archivo_subido = st.file_uploader(
        "Seleccione un archivo PDF o Imagen escaneada (.png, .jpg, .tiff)",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"]
    )

    ruta_a_procesar = None

    if archivo_subido is not None:
        temp_dir = tempfile.mkdtemp()
        ruta_temp = Path(temp_dir) / archivo_subido.name
        with open(ruta_temp, "wb") as f:
            f.write(archivo_subido.getbuffer())
        ruta_a_procesar = ruta_temp
    elif "archivo_demo" in st.session_state and Path(st.session_state["archivo_demo"]).exists():
        ruta_a_procesar = Path(st.session_state["archivo_demo"])
        st.info("📌 Utilizando la planilla sintética de prueba pregenerada.")

    if ruta_a_procesar is None:
        st.warning("👈 Por favor cargue una planilla de auditoría o presione 'Generar y Cargar Muestra Sintética' en la barra lateral.")
        return

    # Pipeline
    with st.spinner("Procesando documento con el pipeline OCR..."):
        try:
            paginas = list(leer_archivo_documento(ruta_a_procesar, dpi=dpi))
            if not paginas:
                st.error("No se pudo extraer ninguna imagen del archivo.")
                return

            num_pag, img_bgr = paginas[0]

            if aplicar_deskew:
                img_procesada, angulo = corregir_inclinacion(img_bgr)
            else:
                img_procesada, angulo = img_bgr, 0.0

            rejilla, celdas = detectar_estructura_tabla(img_procesada)
            img_celdas = resaltar_celdas_detectadas(img_procesada, celdas)

            datos_ocr = procesar_hoja_auditoria(img_procesada, celdas_detectadas=celdas)
            evaluacion = calcular_evaluacion_5s(datos_ocr)

        except Exception as e:
            st.error(f"Error durante el procesamiento: {e}")
            return

    # Paneles
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Diagnóstico y Resultados 5S",
        "📝 Detalle de Criterios",
        "👁️ Visión Artificial y OCR",
        "💾 Exportar Reportes"
    ])

    with tab1:
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Área Auditada", evaluacion.get("area"))
        col_m2.metric("Auditor", evaluacion.get("auditor"))
        col_m3.metric("Fecha", evaluacion.get("fecha"))
        col_m4.metric("Turno", evaluacion.get("turno"))

        st.markdown("---")

        col_g1, col_g2, col_g3, col_g4 = st.columns(4)
        pct = evaluacion.get("porcentaje_global", 0)
        col_g1.metric("Cumplimiento 5S", f"{pct}%")
        col_g2.metric("Puntos Obtenidos", f"{evaluacion.get('puntos_totales')} / {evaluacion.get('puntos_maximos')}")
        col_g3.metric("Ítems Conformes", f"{evaluacion.get('items_conformes')} / {evaluacion.get('total_items')}")
        
        estado = evaluacion.get("estado", "")
        if "EXCELENTE" in estado:
            col_g4.success(f"🟢 {estado}")
        elif "OBSERVACIONES" in estado:
            col_g4.warning(f"🟡 {estado}")
        else:
            col_g4.error(f"🔴 {estado}")

        col_chart1, col_chart2 = st.columns([1, 1])

        with col_chart1:
            st.subheader("Radar de Madurez 5S")
            fig_radar = crear_grafico_radar_5s(evaluacion.get("pilares", {}))
            st.pyplot(fig_radar)

        with col_chart2:
            st.subheader("Cumplimiento por Dimensión")
            for k, p in evaluacion.get("pilares", {}).items():
                st.write(f"**{p['nombre']}** ({p['porcentaje']}%)")
                st.progress(int(p["porcentaje"]))

            st.markdown("### Recomendación Estratégica")
            st.info(evaluacion.get("recomendacion_principal"))

    with tab2:
        st.subheader("Planilla Detallada de Criterios Evaluados")
        df_detalle = pd.DataFrame(evaluacion.get("detalle_items", []))
        if not df_detalle.empty:
            columnas_ver = ["codigo", "pilar", "criterio", "puntuacion", "estado_item", "observaciones"]
            st.dataframe(
                df_detalle[columnas_ver],
                column_config={
                    "codigo": "Código",
                    "pilar": "Pilar",
                    "criterio": "Criterio Evaluado",
                    "puntuacion": st.column_config.NumberColumn("Puntaje (0-4)", format="%d ⭐"),
                    "estado_item": "Estado",
                    "observaciones": "Observaciones"
                },
                use_container_width=True,
                hide_index=True
            )

    with tab3:
        st.subheader("Pipeline de Visión por Computadora")
        col_v1, col_v2 = st.columns(2)
        
        with col_v1:
            st.image(
                cv2.cvtColor(img_procesada, cv2.COLOR_BGR2RGB),
                caption=f"Página Rasterizada (Inclinación ajustada: {angulo:.2f}°)",
                use_container_width=True
            )

        with col_v2:
            st.image(
                cv2.cvtColor(img_celdas, cv2.COLOR_BGR2RGB),
                caption=f"Rejilla y Celdas Detectadas ({len(celdas)} celdas)",
                use_container_width=True
            )

    with tab4:
        st.subheader("Descarga de Reportes y Datos")
        
        ruta_temp_excel = Path("temp_reporte.xlsx")
        exportar_auditoria_excel(evaluacion, ruta_temp_excel)
        with open(ruta_temp_excel, "rb") as f:
            excel_bytes = f.read()

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                label="📥 Descargar Reporte Completo en Excel (.xlsx)",
                data=excel_bytes,
                file_name=f"Reporte_Auditoria_5S_{evaluacion.get('fecha')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        with col_d2:
            json_str = json.dumps(evaluacion, ensure_ascii=False, indent=2)
            st.download_button(
                label="📥 Descargar Diagnóstico en JSON",
                data=json_str,
                file_name=f"Diagnostico_5S_{evaluacion.get('fecha')}.json",
                mime="application/json",
                use_container_width=True
            )


if __name__ == "__main__":
    main()
