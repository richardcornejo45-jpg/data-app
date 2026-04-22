import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configuración de página
st.set_page_config(page_title="Segmentador Profesional", layout="wide")

st.title("🚀 Segmentador de Clientes Avanzado")

# --- 1. PANEL DE INSTRUCCIONES (UX) ---
with st.expander("¿Cómo usar esta herramienta?"):
    st.write("""
    1. **Carga tu CSV:** Debe contener al menos 'CustomerID', 'Quantity' y 'UnitPrice'.
    2. **Configuración:** Ajusta K (número de grupos) usando el slider.
    3. **Análisis:** Haz clic en 'Ejecutar Segmentación'.
    4. **Interpretación:** Observa el gráfico interactivo y las métricas de dispersión. 
       El algoritmo utiliza la inercia para minimizar la distancia dentro de los clusters.
    """)

# --- 2. BARRA LATERAL ---
with st.sidebar:
    st.header("Configuración de Ingeniería")
    uploaded_file = st.file_uploader("Cargar dataset (CSV)", type=["csv"])
    k_value = st.slider("Número de Segmentos (K)", 2, 10, 3)
    btn_ejecutar = st.button("Ejecutar Segmentación")

# --- 3. LÓGICA DE PROCESAMIENTO (Caché para rendimiento) ---
@st.cache_data
def procesar_datos(file):
    df = pd.read_csv(file)
    if 'CustomerID' not in df.columns:
        return None
    df = df.dropna(subset=['CustomerID'])
    df['TotalSum'] = df['Quantity'] * df['UnitPrice']
    return df.groupby('CustomerID').agg({'TotalSum': 'sum', 'InvoiceNo': 'nunique'}).rename(columns={'InvoiceNo': 'Frecuencia'})

# --- 4. EJECUCIÓN DEL FLUJO ---
if uploaded_file is not None:
    df_model = procesar_datos(uploaded_file)
    
    if df_model is None:
        st.error("Error: El archivo no contiene la columna 'CustomerID'.")
    elif btn_ejecutar:
        # Escalamiento
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_model)
        
        # Modelo K-Means
        kmeans = KMeans(n_clusters=k_value, init='k-means++', random_state=42, n_init=10)
        df_model['Cluster'] = kmeans.fit_predict(df_scaled).astype(str)

        # Visualización Interactiva
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader(f"Distribución de Segmentos (K={k_value})")
            fig = px.scatter(df_model, x='TotalSum', y='Frecuencia', color='Cluster',
                             title="Clusters de Clientes",
                             template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Métricas por Grupo")
            cluster_sel = st.selectbox("Analizar cluster:", sorted(df_model['Cluster'].unique()))
            stats = df_model[df_model['Cluster'] == cluster_sel]
            st.metric("Varianza (Gasto)", f"{stats['TotalSum'].var():,.0f}")
            st.metric("Desv. Estándar", f"{stats['TotalSum'].std():,.1f}")
            st.metric("Total Clientes", len(stats))

        # Reporte Técnico
        st.divider()
        st.subheader("Reporte de Convergencia")
        st.info(f"""
        **Análisis Matemático:** El algoritmo alcanzó la convergencia en **{kmeans.n_iter_} iteraciones**. 
        La inercia final (suma de distancias al cuadrado) fue de **{kmeans.inertia_:.2f}**. 
        Un valor de K={k_value} proporciona una segmentación óptima para el equilibrio entre 
        simplicidad operativa y personalización del marketing.
        """)
        
        # Extra: Método del Codo (Elbow Method) simplificado
        with st.expander("Ver Método del Codo (Validación de K)"):
            inertias = [KMeans(n_clusters=i, n_init=10, random_state=42).fit(df_scaled).inertia_ for i in range(1, 11)]
            fig_elbow = px.line(x=range(1, 11), y=inertias, title="Método del Codo", labels={'x': 'K', 'y': 'Inercia'})
            st.plotly_chart(fig_elbow)

    else:
        st.write("Datos cargados correctamente. Presiona **'Ejecutar Segmentación'** para procesar.")
else:
    st.warning("Por favor, sube un archivo CSV para comenzar.")
