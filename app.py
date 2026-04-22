import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configuración mejorada
st.set_page_config(page_title="Segmentador Pro", layout="wide")

st.title("📊 Segmentador de Clientes Avanzado")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("Configuración")
    uploaded_file = st.file_uploader("Cargar dataset (CSV)", type=["csv"])
    k_value = st.slider("Número de Segmentos (K)", 2, 10, 3)
    btn_ejecutar = st.button("Ejecutar Segmentación")

# --- LÓGICA DE PROCESAMIENTO ---
@st.cache_data
def procesar_datos(file):
    df = pd.read_csv(file)
    df = df.dropna(subset=['CustomerID'])
    df['TotalSum'] = df['Quantity'] * df['UnitPrice']
    return df.groupby('CustomerID').agg({'TotalSum': 'sum', 'InvoiceNo': 'nunique'}).rename(columns={'InvoiceNo': 'Frecuencia'})

if uploaded_file is not None:
    df_model = procesar_datos(uploaded_file)
    
    if btn_ejecutar:
        # Escalamiento y Modelo
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_model)
        
        kmeans = KMeans(n_clusters=k_value, init='k-means++', random_state=42, n_init=10)
        df_model['Cluster'] = kmeans.fit_predict(df_scaled).astype(str)

        # Visualización con Plotly (Interactiva)
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig = px.scatter(df_model, x='TotalSum', y='Frecuencia', color='Cluster',
                             title=f"Distribución de Segmentos (K={k_value})",
                             template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Métricas por Grupo")
            cluster_sel = st.selectbox("Analizar cluster:", sorted(df_model['Cluster'].unique()))
            stats = df_model[df_model['Cluster'] == cluster_sel]
            st.metric("Varianza (Gasto)", f"{stats['TotalSum'].var():,.0f}")
            st.metric("Desv. Estándar", f"{stats['TotalSum'].std():,.1f}")

        # Reporte
        st.info(f"**Reporte técnico:** El modelo alcanzó la convergencia en {kmeans.n_iter_} iteraciones, logrando minimizar la inercia a {kmeans.inertia_:.2f}.")
    else:
        st.write("Datos cargados correctamente. Presiona **'Ejecutar Segmentación'** para ver los resultados.")
else:
    st.warning("Por favor, sube un archivo CSV para continuar.")
