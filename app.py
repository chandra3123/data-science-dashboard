# 🌐 Ultimate Interactive Data Science Dashboard
# ---------------------------------------------
# To run: streamlit run app.py
# Make sure to install:
# pip install streamlit pandas numpy seaborn matplotlib scikit-learn plotly

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --------------- PAGE CONFIG -------------------
st.set_page_config(page_title="Data Science Dashboard", layout="wide", page_icon="🧠")

# --------------- HEADER SECTION ----------------
st.markdown("""
<style>
.big-font { font-size:38px !important; font-weight:700; color:#4BB543; text-align:center; }
.sub-font { font-size:20px; text-align:center; color:#555; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">🧠 Advanced Data Science Visualization Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-font">An interactive platform for exploring, visualizing, and analyzing datasets with PCA, Clustering, and Dynamic Charts</p>', unsafe_allow_html=True)
st.write("---")

# --------------- LOAD DEFAULT OR UPLOADED DATA -------------------
# Try to load default dataset
default_data_available = False
try:
    default_df = pd.read_csv("complete.csv")
    default_data_available = True
except:
    default_df = None

# File uploader
st.subheader("📂 Data Source")
col1, col2 = st.columns([2, 1])

with col1:
    uploaded = st.file_uploader("Upload your own CSV file (optional)", type=["csv"])

with col2:
    if default_data_available:
        use_default = st.button("🔄 Use Sample COVID-19 Dataset", type="primary")
    else:
        use_default = False

# Determine which dataset to use
df = None
if uploaded:
    df = pd.read_csv(uploaded)
    st.success("✅ Your dataset uploaded successfully!")
elif use_default or (default_data_available and uploaded is None):
    df = default_df
    st.info("📊 Using sample COVID-19 dataset (India, Jan-Apr 2020) - 4,692 records")
    
if df is not None:
    st.dataframe(df.head(10), use_container_width=True)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # SIDEBAR
    st.sidebar.header("⚙️ Dashboard Controls")
    show_summary = st.sidebar.checkbox("Show Summary Statistics", True)
    show_cleaning = st.sidebar.checkbox("Clean Missing Values & Outliers", True)
    enable_pca = st.sidebar.checkbox("Enable PCA & Clustering", True)
    theme = st.sidebar.radio("Theme", ["Light", "Dark"])
    
    if theme == "Dark":
        st.markdown('<style>body { background-color: #0E1117; color: white; }</style>', unsafe_allow_html=True)
    
    # ---------------- SUMMARY -------------------
    if show_summary:
        st.subheader("📋 Dataset Summary")
        st.write(df.describe())
        st.write("### Missing Values")
        st.bar_chart(df.isnull().sum())
    
    # ---------------- CLEANING -------------------
    if show_cleaning:
        for col in df.columns:
            if df[col].dtype == "O":
                if len(df[col].mode()) > 0:
                    df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
        for col in numeric_cols:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            df[col] = np.clip(df[col], Q1 - 1.5*IQR, Q3 + 1.5*IQR)
        st.success("🧹 Data cleaned successfully (missing values & outliers handled).")
    
    # ---------------- VISUALIZATION SECTION ----   ---------------
    st.header("📊 Data Visualization")
    
    col1, col2 = st.columns(2)
    with col1:
        if len(numeric_cols) > 0:
            selected_feature = st.selectbox("Select feature for Histogram", numeric_cols)
            fig = px.histogram(df, x=selected_feature, nbins=30, title=f"Distribution of {selected_feature}", color_discrete_sequence=["#2E86C1"])
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if len(numeric_cols) >= 2:
            x_feat = st.selectbox("X-axis for Scatter", numeric_cols, index=0)
            y_feat = st.selectbox("Y-axis for Scatter", numeric_cols, index=1)
            fig2 = px.scatter(df, x=x_feat, y=y_feat, color_discrete_sequence=["#00CC96"], title=f"{x_feat} vs {y_feat}")
            st.plotly_chart(fig2, use_container_width=True)
    
    if len(numeric_cols) >= 2:
        st.subheader("🔥 Correlation Heatmap")
        corr = df[numeric_cols].corr()
        fig3 = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu", title="Feature Correlation Matrix")
        st.plotly_chart(fig3, use_container_width=True)
    
    # ---------------- PCA & CLUSTERING -------------------
    if enable_pca and len(numeric_cols) > 2:
        st.header("🧩 PCA & K-Means Clustering")
        n_comp = st.slider("Select number of PCA components", 2, min(5, len(numeric_cols)), 3)
        
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df[numeric_cols])
        
        pca = PCA(n_components=n_comp)
        pca_data = pca.fit_transform(scaled)
        explained = np.sum(pca.explained_variance_ratio_)*100
        
        st.write(f"✅ PCA completed — Total Variance Captured: **{explained:.2f}%**")
        pca_df = pd.DataFrame(pca_data, columns=[f"PC{i+1}" for i in range(n_comp)])
        
        # Optimal clusters using silhouette
        best_k, best_score = 0, -1
        for k in range(2,6):
            km = KMeans(n_clusters=k, random_state=42)
            labels = km.fit_predict(pca_df)
            score = silhouette_score(pca_df, labels)
            if score > best_score:
                best_k, best_score = k, score
        
        st.write(f"📊 Optimal Clusters Found: **{best_k}** (Silhouette Score: {best_score:.3f})")
        kmeans = KMeans(n_clusters=best_k, random_state=42)
        pca_df["Cluster"] = kmeans.fit_predict(pca_df)
        
        # 3D Plot if 3 components
        if n_comp >= 3:
            fig4 = px.scatter_3d(pca_df, x="PC1", y="PC2", z="PC3", color=pca_df["Cluster"].astype(str),
                                 title="3D PCA Cluster Visualization", color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig4, use_container_width=True)
        else:
            fig5 = px.scatter(pca_df, x="PC1", y="PC2", color=pca_df["Cluster"].astype(str),
                              title="2D PCA Cluster Visualization", color_discrete_sequence=px.colors.qualitative.Bold)
            st.plotly_chart(fig5, use_container_width=True)
        
        # PCA Variance Graph
        fig6 = go.Figure(data=[go.Bar(
            x=[f"PC{i+1}" for i in range(n_comp)],
            y=pca.explained_variance_ratio_*100,
            marker_color='teal'
        )])
        fig6.update_layout(title="PCA Component Variance (%)", xaxis_title="Principal Components", yaxis_title="Variance (%)")
        st.plotly_chart(fig6, use_container_width=True)
    
    # ---------------- INSIGHTS -------------------
    st.write("---")
    st.header("💡 Insights & Key Findings")
    st.markdown("""
    - **PCA** helps in reducing dimensions while keeping maximum variance.
    - **K-Means Clustering** groups similar data points for easier segmentation.
    - Use this dashboard for **COVID trend analysis, sales insights, population behavior,** or any dataset exploration.
    - Try adjusting PCA components and cluster count to explore hidden patterns dynamically.
    """)

else:
    # Show welcome message when no data is loaded
    st.info("👆 Upload your CSV file or click 'Use Sample COVID-19 Dataset' to get started!")
    
    # Show sample dataset info if available
    if default_data_available:
        st.markdown("---")
        st.subheader("📊 About the Sample Dataset")
        st.markdown("""
        The sample COVID-19 dataset includes:
        - **4,692 records** from India
        - **36 states/union territories**
        - **Time period**: January 30 - April 27, 2020
        - **Features**: Date, Location, Cases, Deaths, Recovered, and more
        
        Click **'Use Sample COVID-19 Dataset'** above to explore it!
        """)
    
    st.markdown("---")
    st.subheader("✨ Dashboard Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 Visualizations**")
        st.markdown("- Interactive histograms")
        st.markdown("- Scatter plots")
        st.markdown("- Correlation heatmaps")
    
    with col2:
        st.markdown("**🧩 Machine Learning**")
        st.markdown("- PCA analysis")
        st.markdown("- K-Means clustering")
        st.markdown("- 3D visualizations")
    
    with col3:
        st.markdown("**🎨 Customization**")
        st.markdown("- Light/Dark themes")
        st.markdown("- Data cleaning options")
        st.markdown("- Interactive controls")

