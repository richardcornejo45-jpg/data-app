import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página
st.set_page_config(page_title="Data App: Segmentación Retail", layout="wide")

st.title("🚀 Segmentador de Clientes en Tiempo Real")
st.markdown("""
Esta aplicación permite cargar datos de ventas y agrupar clientes utilizando el algoritmo **K-Means**.
""")

# --- BARRA LATERAL (CONFIGURACIÓN) ---
st.sidebar.header("Configuración de Ingeniería")
uploaded_file = st.sidebar.file_uploader("Carga tu archivo CSV", type=["csv"])
k_value = st.sidebar.slider("Selecciona el valor de K (Clusters)", min_value=2, max_value=10, value=3)

if uploaded_file is not None:
    # 1. PREPARACIÓN (Carga y Limpieza)
    df = pd.read_csv(uploaded_file)
    
    # Procesamiento rápido para el dataset de Online Retail
    if 'CustomerID' in df.columns:
        df = df.dropna(subset=['CustomerID'])
        if 'Quantity' in df.columns and 'UnitPrice' in df.columns:
            df['TotalSum'] = df['Quantity'] * df['UnitPrice']
            
        # Agrupamos por cliente
        df_model = df.groupby('CustomerID').agg({
            'TotalSum': 'sum',
            'InvoiceNo': 'nunique'
        }).rename(columns={'InvoiceNo': 'Frecuencia'})

        # 2. VALIDACIÓN (Escalamiento)
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_model)

        # 3. DESCUBRIMIENTO (K-Means)
        kmeans = KMeans(n_clusters=k_value, init='k-means++', random_state=42, n_init=10)
        df_model['Cluster'] = kmeans.fit_predict(df_scaled)

        # --- VISUALIZACIÓN ---
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader(f"Visualización de Grupos (K={k_value})")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=df_model, x='TotalSum', y='Frecuencia', hue='Cluster', palette='viridis', ax=ax)
            
            # Dibujar centroides
            centroids = scaler.inverse_transform(kmeans.cluster_centers_)
            ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroides')
            plt.legend()
            st.pyplot(fig)

        with col2:
            st.subheader("Métricas de Dispersión")
            cluster_sel = st.selectbox("Selecciona un Cluster para analizar:", sorted(df_model['Cluster'].unique()))
            df_cluster = df_model[df_model['Cluster'] == cluster_sel]
            
            st.metric("Varianza del Gasto", f"{df_cluster['TotalSum'].var():,.2f}")
            st.metric("Desviación Estándar", f"{df_cluster['TotalSum'].std():,.2f}")
            st.metric("Total Clientes", len(df_cluster))

        # --- REPORTE DE CONVERGENCIA ---
        st.divider()
        st.subheader("Reporte de Convergencia")
        st.write(f"""
        Para el valor de **K={k_value}**, el algoritmo alcanzó la convergencia tras {kmeans.n_iter_} iteraciones. 
        En este punto, los centroides (marcados con una X roja) se estabilizaron en el centro geométrico de cada grupo, 
        minimizando la inercia interna. Esto significa que la separación de clientes es la más óptima posible para la 
        distribución actual de los datos.
        """)
    else:
        st.error("El archivo no tiene el formato esperado. Asegúrate de que contenga 'CustomerID'.")
else:
    st.info("A la espera de que cargues un archivo CSV para comenzar el análisis.")
