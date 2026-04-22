import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go # Necesario para dibujar los centroides
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Segmentador Pro", layout="wide")
st.title("🚀 Segmentador de Clientes Avanzado")

# --- 1. CONFIGURACIÓN ---
with st.sidebar:
    st.header("Configuración")
    uploaded_file = st.file_uploader("Cargar dataset (CSV)", type=["csv"])
    
    if uploaded_file:
        df_temp = pd.read_csv(uploaded_file)
        # Selección de columnas para el usuario (Flexibilidad extra)
        cols = st.multiselect("Selecciona columnas para segmentar", df_temp.select_dtypes(include='number').columns.tolist(), default=['Quantity', 'UnitPrice'])
        k_value = st.slider("Número de Segmentos (K)", 2, 10, 3)
        btn_ejecutar = st.button("Ejecutar Segmentación")

# --- 2. LÓGICA ---
if uploaded_file is not None and 'btn_ejecutar' in locals() and btn_ejecutar:
    df = pd.read_csv(uploaded_file).dropna()
    
    # Preprocesamiento
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df[cols])
    
    # Modelo
    kmeans = KMeans(n_clusters=k_value, init='k-means++', random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(data_scaled).astype(str)
    
    # Obtener Centroides y revertir el escalado
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # --- 3. VISUALIZACIÓN CON "X" (Centroides) ---
    fig = px.scatter(df, x=cols[0], y=cols[1], color='Cluster', title="Clusters y Centroides")
    
    # Añadir los centroides como una capa de "X" roja
    fig.add_trace(go.Scatter(
        x=centroids[:, 0], y=centroids[:, 1],
        mode='markers',
        marker=dict(size=15, symbol='x', color='red', line=dict(width=2, color='white')),
        name='Centroides'
    ))
    
    st.plotly_chart(fig, use_container_width=True)

    # --- 4. MÉTRICAS ---
    st.subheader("Métricas del Modelo")
    st.info(f"Convergencia alcanzada en {kmeans.n_iter_} iteraciones. Inercia: {kmeans.inertia_:.2f}")

elif uploaded_file is None:
    st.warning("Por favor sube un archivo CSV.")
