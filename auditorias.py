
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
