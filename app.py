import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configuración de página
st.set_page_config(page_title="Segmentador Profesional", layout="wide")
st.title("🚀 Segmentador de Clientes Avanzado")

# --- 1. BARRA LATERAL ---
with st.sidebar:
    st.header("Configuración de Datos")
    uploaded_file = st.file_uploader("Cargar dataset (CSV)", type=["csv"])
    
    # Solo mostramos controles si hay archivo
    if uploaded_file is not None:
        # Forzar lectura al inicio
        uploaded_file.seek(0)
        df_base = pd.read_csv(uploaded_file)
        
        # Selección de columnas numéricas para el análisis
        numeric_cols = df_base.select_dtypes(include='number').columns.tolist()
        cols_to_use = st.multiselect("Selecciona columnas para segmentar", numeric_cols, default=numeric_cols[:2])
        
        k_value = st.slider("Número de Segmentos (K)", 2, 10, 3)
        btn_ejecutar = st.button("Ejecutar Segmentación")

# --- 2. LÓGICA DE PROCESAMIENTO ---
if uploaded_file is not None and 'btn_ejecutar' in locals() and btn_ejecutar:
    if not cols_to_use:
        st.warning("Por favor, selecciona al menos una columna.")
    else:
        # Limpieza y preparación
        df = df_base.dropna(subset=cols_to_use)
        
        # Escalamiento
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df[cols_to_use])
        
        # Modelo K-Means
        kmeans = KMeans(n_clusters=k_value, init='k-means++', random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(data_scaled).astype(str)
        
        # Obtener Centroides y revertir el escalado
        centroids = scaler.inverse_transform(kmeans.cluster_centers_)
        
        # --- 3. VISUALIZACIÓN ---
        st.subheader(f"Distribución: {cols_to_use[0]} vs {cols_to_use[1]}")
        
        # Gráfico principal
        fig = px.scatter(df, x=cols_to_use[0], y=cols_to_use[1], color='Cluster', 
                         title="Clusters y Centroides (X)", template="plotly_white")
        
        # Añadir las "X" rojas de los centroides
        fig.add_trace(go.Scatter(
            x=centroids[:, 0], y=centroids[:, 1],
            mode='markers',
            marker=dict(size=18, symbol='x', color='red', line=dict(width=3, color='black')),
            name='Centroides'
        ))
        
        st.plotly_chart(fig, use_container_width=True)

        # --- 4. MÉTRICAS ---
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Iteraciones", kmeans.n_iter_)
        with col_m2:
            st.metric("Inercia (Error)", f"{kmeans.inertia_:.2f}")

        st.info("El algoritmo ha convergido. Los puntos marcados con **X** roja representan los centros geométricos de cada segmento.")

elif uploaded_file is None:
    st.info("👈 Por favor, carga un archivo CSV en la barra lateral para comenzar.")
