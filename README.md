# App-Danu

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import os
import requests
import tempfile
import requests





# === CARGA DE MODELO Y ENCODER ===
modelo = joblib.load("modelo_dias_pipeline_3clases.sav")
le = joblib.load("label_encoder_dias_3clases.sav")

if "df" not in st.session_state:
    df = pd.read_csv('df_equipo5.csv')
    df['orden_compra_timestamp'] = pd.to_datetime(df['orden_compra_timestamp'], errors='coerce')
    if 'estado' not in df.columns:
        df['estado'] = 'pendiente de despacho'
    st.session_state.df = df
else:
    df = st.session_state.df


# === COLUMNAS USADAS PARA PREDICCIÓN ===
columnas_modelo = ['categoria', '#_deproductos', 'total_peso_g', 'precio', 'costo_de_flete',
                   'distancia_km', 'velocidad_kmh', 'duracion_estimada_min', 'region', 'dc_asignado',
                   'es_feriado', 'es_fin_de_semana', 'dias_promedio_ciudad',
                   'hora_compra', 'nombre_dia', 'mes', 'año',
                   'traffic', 'area',]

# === SIMULADOR DE PEDIDOS ===
if 'index_pedido' not in st.session_state:
    st.session_state.index_pedido = 0

# Mostrar logo en el sidebar
logo = Image.open("danu_logo.png")
st.sidebar.image(logo, use_container_width=True)


# === FILTROS ===
regiones = df['nombre_dc'].dropna().unique()
categorias = df['categoria'].dropna().unique()
region_sel = st.sidebar.selectbox("Elegir Centro Distribución", np.append(["Todas"], sorted(regiones)))
categoria_sel = st.sidebar.selectbox("Elegir Categoria", np.append(["Todas"], sorted(categorias)))



df_filtrado = df.copy()
if region_sel != "Todas":
    df_filtrado = df_filtrado[df_filtrado["nombre_dc"] == region_sel]
if categoria_sel != "Todas":
    df_filtrado = df_filtrado[df_filtrado["categoria"] == categoria_sel]


tab1, tab2 = st.tabs(["📦 Seguimiento y asistente", "📊 KPIs y Reporte"])

with tab1:
    if len(df_filtrado) == 0:
        st.warning("⚠ No hay pedidos disponibles con los filtros actuales.")
    else:
    # Corrige índice si está fuera de rango
        if st.session_state.index_pedido >= len(df_filtrado):
            st.session_state.index_pedido = 0  # Reiniciar índice si es mayor al total

        index_final = st.session_state.index_pedido
        pedido_actual = df_filtrado.iloc[index_final]

        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            if st.button("Siguiente pedido"):
                if st.session_state.index_pedido + 1 < len(df_filtrado):
                    st.session_state.index_pedido += 1

        with col2:
            estado_actual = pedido_actual.get("estado", "pendiente de despacho")
            index_real = df_filtrado.index[st.session_state.index_pedido]

            if estado_actual != "despachado":
                if st.button("Marcar como despachado"):
                    st.session_state.df.at[index_real, 'estado'] = 'despachado'
                    st.success("✅ Pedido marcado como despachado.")
                    st.rerun()
            else:
                st.info("✅ Este pedido ya fue marcado como despachado.")



        with col3:
            if 'busqueda_id' not in st.session_state:
                st.session_state.busqueda_id = ""

            nuevo_id = st.text_input("🔍 Buscar pedido por ID", placeholder="Escribe el ID completo del pedido",
                                     value=st.session_state.busqueda_id, key="input_busqueda_id")

            if nuevo_id != st.session_state.busqueda_id:
                st.session_state.busqueda_id = nuevo_id

                if nuevo_id in df_filtrado['order_id'].astype(str).values:
                    index_encontrado = df_filtrado[df_filtrado['order_id'].astype(str) == nuevo_id].index[0]
                    st.session_state.index_pedido = df_filtrado.index.get_loc(index_encontrado)
                    st.rerun()  # Solo se ejecuta si realmente hubo un cambio válido
                elif nuevo_id != "":
                    st.warning("❌ El ID ingresado no existe en los pedidos filtrados.")



        pedido_df = pd.DataFrame([pedido_actual[columnas_modelo]])

        # === PREDICCIÓN ===
        y_pred_encoded = modelo.predict(pedido_df)[0]
        y_pred_label = le.inverse_transform([y_pred_encoded])[0]

        # === INTERFAZ PRINCIPAL ===
        st.title("Reporte Pedidos con Predicción")
        st.markdown(f"### Pedido #{pedido_actual['order_id']}")
        st.write(f"**Ciudad:** {pedido_actual['ciudad_cliente']}")
        st.write(f"**categoria:** {pedido_actual['categoria']}")
        st.write(f"**Peso (g):** {pedido_actual['total_peso_g']}")
        st.write(f"**Productos:** {pedido_actual['#_deproductos']}")
        st.write(f"**Distancia estimada:** {pedido_actual['distancia_km']} km")
        st.write(f"**Estado actual del pedido:** {estado_actual}")

        st.write("---")
        st.subheader("Predicción")
        if y_pred_label == "1-5 días":
            st.error(f"⏳ Tiempo estimado de entrega: **{y_pred_label}**")
        elif y_pred_label == "6-10 días":
            st.warning(f"🕐 Tiempo estimado de entrega: **{y_pred_label}**")
        else:
            st.success(f"📦 Tiempo estimado de entrega: **{y_pred_label}**")

        
        
        # === SISTEMA DE RECOMENDACIÓN 1: Visual (para un pedido individual) ===
        def recomendacion_pedido_individual(p):
        # Regla 1: Predicción urgente y ciudad lenta
            if p['prediccion'] == '1-5 días' and p['dias_promedio_ciudad'] > 6:
                return f"⚠ El pedido a {p['ciudad_cliente']} suele tardar más de {p['dias_promedio_ciudad']} días. Urge salida hoy."

        # Regla 2: Larga distancia
            elif p['distancia_km'] > 1200:
                return f"🚛 Pedido de larga distancia ({p['distancia_km']} km). Considera salida en ruta especial o preferente."

        # Regla 3: Fin de semana + rural
            elif p['es_fin_de_semana'] == 1 and str(p['area']).lower() == 'rural':
                return f"📍 Zona rural y fin de semana. Mayor probabilidad de retraso si no se despacha pronto."

        # Regla 4: Categoría delicada
            elif str(p['categoria']).lower() in ['floreria', 'alimentos y bebidas', 'arte y manualidades', 'electronica y tecnología']:
                return f"📦 Categoría delicada ({p['categoria']}). Se sugiere manejo prioritario para evitar devoluciones."

        # Regla 5: Región y predicción largas
            elif str(p['region']).lower() in ['norte', 'sureste', 'noroeste'] and p['prediccion'] == '+10 días':
                return f"⏳ Entrega extendida a {p['region']}. Puede ser elegible para consolidación en ruta agrupada."

        # Regla 6: Pocos productos pero gran distancia
            elif p['#_deproductos'] == 1 and p['distancia_km'] > 300:
                return f"📦 Pedido individual con gran distancia. Evalúa si conviene consolidar con más paquetes o enviar directo."

        # Regla 7: Predicción moderada y ciudad conocida
            elif p['prediccion'] == '6-10 días' and p['dias_promedio_ciudad'] <= 6:
                return f"🕐 Tiempo moderado pero la ciudad suele ser rápida. Considera adelantar si hay espacio."
            
        # Regla 8: Meses de alta demanda
            elif p['mes'] in [5, 11, 12]:
                if p['prediccion'] in ['6-10 días', '+10 días']:
                    return f"🎄 Pedido en temporada alta. Considera adelantar salida para evitar congestión logística."
                else:
                    return f"📆 Aunque la entrega es rápida, suele tener alta carga de pedidos debido a temporada alta. Monitorea cumplimiento."

        # Regla 9: Día feriado cerca (esto si ya usas 'es_feriado')
            elif p['es_feriado'] == 1:
                return "📅 Pedido realizado en día feriado. Revisa si afectará tiempos de salida o tránsito."
            
            elif str(p['categoria']).lower() == 'hogar y muebles':
                if p['prediccion'] == '+10 días':
                    return f"🪑 Pedido de muebles con entrega prolongada. Verifica capacidad de almacenamiento y planea salida temprana."
                elif p['total_peso_g'] > 20000:
                    return f"📦 Pedido pesado de hogar/muebles. Considera envío directo para evitar manipulación innecesaria."
                else:
                    return f"🏠 Categoría voluminosa. Evalúa ruta con menos paradas para reducir riesgos de daño."



        # Default: ya revisado
            else:
                return "✅ Sin alertas. El pedido puede ser planificado con normalidad."


            # === SISTEMA DE RECOMENDACIÓN 2: CSV/Reporte (más robusto) ===
        def recomendacion_reporte_csv(row):
            pred = row['prediccion']
            riesgo = row.get('riesgo_historico', 'NORMAL')
            ciudad = row['ciudad_cliente']
            cat = row['categoria']
            mediana_dias = row.get('dias_median_categoria_ciudad', 10)

            if pred == "1-5 días":
                if riesgo == "ALTO":
                    return f"🚨 Entrega rápida prevista, pero riesgo histórico ALTO. Asegurar salida inmediata hacia {ciudad}."
                else:
                    return f"✅ Predicción favorable. Planificar envío inmediato hacia {ciudad} en categoría {cat}."

            elif pred == "6-10 días":
                if riesgo == "ALTO":
                    return f"⚠ Pedido con ventana moderada, pero historial ALTO en {ciudad} para {cat}. Evaluar ruta preferente."
                else:
                    return f"🕐 Envío dentro de rango esperado. Se recomienda despacho en próximas rutas."

            elif pred == "10+":
                if str(cat).lower() in ['hogar y muebles']:
                    return f"🛏 Categoría {cat} con entrega tardía. Suele ocupar volumen alto, evalúa enviar solo si hay carga suficiente."
                elif riesgo == "ALTO":
                    return f"🚨 Predicción larga y riesgo ALTO. Considera priorizar o reagendar con mejores condiciones."
                elif str(row['region']).lower() in ['norte', 'sureste'] and row['distancia_km'] > 1000:
                    return f"🚚 Pedido al norte con más de 1000 km. Puede agruparse con otras rutas semanales para optimizar transporte."
                elif row['mes'] in [11, 12, 1, 5]:  # Temporada alta
                    return f"📦 Pedido en temporada alta. Aunque tiene margen, despacharlo pronto evita saturaciones."
                elif mediana_dias <= 10:
                    return f"❗ Tiempo estimado supera la mediana histórica ({mediana_dias} días) para {cat} en {ciudad}. Revisa alternativas."
                else:
                    return f"⏳ Pedido puede esperar. Recomendado para consolidación o despacho en días de baja operación."


        # Agrega la predicción al pedido_actual
        pedido_actual['prediccion'] = y_pred_label

        # Mostrar recomendación
        st.markdown("### Recomendación del sistema")
        st.info(recomendacion_pedido_individual(pedido_actual))
        


        st.markdown("---")
        st.header("🤖 Asistente AI - Logístico")

        # Inicializa historial
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Columna doble: resumen a la izquierda, chat a la derecha
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### 📦 Detalles del Pedido Actual")
            st.markdown(f"**ID:** {pedido_actual['order_id']}")
            st.markdown(f"**Ciudad:** {pedido_actual['ciudad_cliente']}")
            st.markdown(f"**Categoría:** {pedido_actual['categoria']}")
            st.markdown(f"**Peso:** {pedido_actual['total_peso_g']}g")
            st.markdown(f"**Distancia:** {pedido_actual['distancia_km']} km")
            st.markdown(f"**Zona:** {pedido_actual['area']}")
            st.markdown(f"**Predicción ML:** {y_pred_label}")
            st.markdown(f"**Estado actual:** {pedido_actual.get('estado', 'No definido')}")
            st.markdown(f"**¿Feriado?:** {'Sí' if pedido_actual['es_feriado'] == 1 else 'No'}")
            st.markdown(f"**Mes:** {pedido_actual['mes']}")

        with col2:
            st.markdown("### 💬 Chatea con tu asistente")

            # Mostrar historial
            for turno in st.session_state.chat_history[-5:]:  # Últimos 5 mensajes
                st.markdown(f"""
                <div style="background-color:#e1f5fe;padding:12px;border-radius:12px;margin-bottom:5px;width:fit-content">
                <b>🧍 Tú:</b><br>{turno['pregunta']}
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style="background-color:#f0f0f0;padding:12px;border-radius:12px;margin-bottom:20px;width:fit-content">
                <b>🤖 Asistente:</b><br>{turno['respuesta']}
                </div>
                """, unsafe_allow_html=True)

            # Input nuevo
            user_input = st.chat_input("Haz una pregunta sobre este pedido...")

            if user_input:
                # Buscar históricos similares
                similares = df[
                    (df['ciudad_cliente'] == pedido_actual['ciudad_cliente']) &
                    (df['categoria'] == pedido_actual['categoria']) &
                    (df['total_peso_g'].between(pedido_actual['total_peso_g'] - 500, pedido_actual['total_peso_g'] + 500))
                ]

                mediana = similares['dias_entrega'].median() if not similares.empty else "N/A"
                proporcion_larga = (similares['dias_entrega'] > 5).mean() if not similares.empty else 0

                # PROMPT RESUMIDO Y PROFESIONAL
                prompt = f"""
        Eres un asistente experto en logística de última milla. Analiza el siguiente pedido y responde en tres bloques bien separados:

        Condiciones:
        - Cuales son las condiciones del destino
        - A cuantos kilometros esta, que tanto trafico hay, etc

        📊 Historial:
        - Indica cuántos pedidos similares hay (misma ciudad, categoría, peso similar).
        - ¿Cuál es la mediana de días de entrega?
        - ¿Qué porcentaje superó los 5 días?

        📦 Estimación:
        - Según la predicción del modelo ML y según pedidos historicos cuanto prevees que tarde ¿cuántos días podría tardar este pedido?

        ✅ Recomendación:
        - Da 1 a 3 acciones operativas concretas que aporten mucho a la toma de decision
        y que ayuden al encargado logístico tomar decisiones informadas para asegurar que el pedido llegue a tiempo y
        se mantenga la eficiencia dentro del centro de distribución.

        Usa solo los datos que se proporcionan. No inventes otros.

        ---
        📦 Datos del pedido:
        - ID: {pedido_actual['order_id']}
        - Ciudad: {pedido_actual['ciudad_cliente']}
        - Categoría: {pedido_actual['categoria']}
        - Peso: {pedido_actual['total_peso_g']}g
        - Productos: {pedido_actual['#_deproductos']}
        - Distancia estimada: {pedido_actual['distancia_km']} km
        - Zona: {pedido_actual['area']}
        - Mes: {pedido_actual['mes']}
        - Estado actual: {pedido_actual.get('estado', 'No definido')}
        - Predicción ML: {y_pred_label}

        📊 Histórico:
        - {len(similares)} pedidos similares encontrados
        - Mediana de días: {mediana}
        - % con más de 5 días: {proporcion_larga:.0%}

        ---
        Pregunta del operador:
        {user_input}
        """


                # Llamar al modelo (Ollama u otro)
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": "llama3.2:latest", "prompt": prompt, "stream": False}
                )

                if response.status_code == 200:
                    respuesta_ia = response.json()["response"]

                    # Guardar en historial
                    st.session_state.chat_history.append({
                        "pregunta": user_input,
                        "respuesta": respuesta_ia
                    })

                    st.rerun()
                else:
                    st.error("❌ Error al generar respuesta del modelo IA.")



with tab2:
    ### Generacion de Reporte
    st.subheader("Generar reporte de pedidos urgentes por fecha")

    # Selector de fecha
    if "fecha_reporte" not in st.session_state:
        st.session_state.fecha_reporte = pd.to_datetime("2018-01-16").date()

    fecha_elegida = st.date_input("Selecciona una fecha para el reporte", value=st.session_state.fecha_reporte)

    # Guarda el nuevo valor si cambia
    if fecha_elegida != st.session_state.fecha_reporte:
        st.session_state.fecha_reporte = fecha_elegida

    # Función para aplicar predicciones por fila
    @st.cache_data
    def generar_predicciones(df):
        df = df.copy()
        X = df[columnas_modelo]
        encoded_preds = modelo.predict(X)
        df['prediccion'] = le.inverse_transform(encoded_preds)
        return df


    # Aplica predicción al DataFrame filtrado
    df_con_pred = generar_predicciones(df_filtrado)
    df_con_pred['recomendacion'] = df_con_pred.apply(recomendacion_reporte_csv, axis=1)


    # Filtra por fecha elegida y pedidos urgentes
    df_reporte = df_con_pred[
        (df_con_pred['orden_compra_timestamp'].dt.date == st.session_state.fecha_reporte) 
     #   (df_con_pred['prediccion'] == "1-5 días")
    ]

    # Mostrar KPIs antes del reporte
    df_kpis = df.copy()

    # Aplicar los mismos filtros del sidebar
    if region_sel != "Todas":
        df_kpis = df_kpis[df_kpis["nombre_dc"] == region_sel]
    if categoria_sel != "Todas":
        df_kpis = df_kpis[df_kpis["categoria"] == categoria_sel]

    # Filtrar por fecha
    df_kpis = df_kpis[df_kpis['orden_compra_timestamp'].dt.date == fecha_elegida]

    if not df_kpis.empty:
        df_kpis_pred = generar_predicciones(df_kpis)
        total_pedidos = len(df_kpis_pred)
        urgentes = (df_kpis_pred['prediccion'] == "1-5 días").sum()
        moderados = (df_kpis_pred['prediccion'] == "6-10 días").sum()
        extensos = (df_kpis_pred['prediccion'].str.contains("10+")).sum()

        st.markdown("### 📊 KPIs de pedidos para la fecha seleccionada y filtros aplicados:")
        
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("📦 Total de pedidos", total_pedidos)
        col2.metric("⚠️ Urgentes (1-5 días)", urgentes)
        col3.metric("🕐 Moderados (6-10 días)", moderados)
        col4.metric("⏳ Largos (+10 días)", extensos)
        
        # === ALERTAS OPERATIVAS ===
        st.markdown("### 🔔 Alertas operativas")

        # 1. Muchos pedidos urgentes
        if urgentes > 50:
            st.warning(f"⚠ Atención: {urgentes} pedidos urgentes programados para el {fecha_elegida}. Considera rutas adicionales o refuerzos.")

        # 3. Día feriado con actividad
        if df_kpis_pred['es_feriado'].sum() > 0 and total_pedidos > 30:
            st.warning("📅 Pedidos programados para día feriado. Confirma disponibilidad de transportistas y personal de carga.")

        # 4. Categorías delicadas con urgencia
        cat_urgente = df_kpis_pred[
            (df_kpis_pred['categoria'].str.lower().isin(['hogar y muebles', 'alimentos y bebidas', 'floreria',
                                                         'arte y manualidades',
                                                         'electronica y tecnología'])) &
            (df_kpis_pred['prediccion'] == "1-5 días")]
        if not cat_urgente.empty:
            st.info(f"🍽 Hay {len(cat_urgente)} pedidos urgentes de categoría sensible. Requieren manejo especial.")

        # 5. Saturación general
        if total_pedidos > 200:
            st.error("🚨 Alta carga operativa prevista. Se recomienda coordinar turnos y disponibilidad de rutas adicionales.")
        
        else:
            st.info("No hay alertas específicas para este día")
        
    else:
        st.info("❌ No hay pedidos con los filtros y fecha seleccionados.")


    # Mostrar resultados
    st.markdown(f"### Reporte de pedidos urgentes del {fecha_elegida}")
    if not df_reporte.empty:
        st.dataframe(df_reporte[['order_id', 'ciudad_cliente', 'categoria', 'nombre_dc', 'prediccion', 'recomendacion', 'estado']])
        
        # Botón para descargar CSV
        columnas_csv = ['order_id','ciudad_cliente','categoria','nombre_dc','orden_compra_timestamp','prediccion',
                        'dias_median_categoria_ciudad','recomendacion','estado']
        
        csv = df_reporte[columnas_csv].to_csv(index=False).encode('utf-8')
        st.download_button("📥 Descargar reporte CSV", data=csv, file_name=f"reporte_urgentes_{fecha_elegida}.csv")
            

        destinatario = st.text_input("Correo del destinatario", placeholder="gerente@danulogistica.com")

        if st.button("📧 Enviar reporte"):
            response = send_report_mailgun(df_reporte[columnas_csv], destinatario)

            if response.status_code == 200:
                st.success("📤 Reporte enviado exitosamente al encargado del centro de distribución.")
            else:
                st.error(f"❌ Error al enviar el reporte: {response.status_code}")
                
    else:
        st.success("📬 No se registraron pedidos urgentes en esta fecha. Buen desempeño logístico!")

            
        
