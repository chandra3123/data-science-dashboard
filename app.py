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

st.markdown('<p class="big-font">  DATA SCIENCE ASSIGNMENT </p>', unsafe_allow_html=True)


# Team Information
st.markdown("""
<div style='text-align: center; background: white; border: 2px solid black; padding: 15px; border-radius: 10px; margin: 20px 0;'>
    <p style='color: black; font-size: 18px; font-weight: 600; margin: 0;'>OUR TEAM  | ECE-I Section</p>
    <p style='color: black; font-size: 14px; margin: 5px 0 0 0;'>
        1. S.V. Chirudeep Reddy (Roll No: 64) &nbsp;|&nbsp; 2. U. Ganesh (Roll No: 57) &nbsp;|&nbsp; 3. Kyathendra Venkata Surya (Roll No: 11) &nbsp;|&nbsp; 4. Dev Rehanth (Roll No: 55)
    </p>
</div>
""", unsafe_allow_html=True)

# Assignment Question
with st.expander("📚 Assignment Details - Click to View", expanded=False):
    st.markdown("""
    ### 📋 Assignment Questions:
    
    **1. Data Pre-processing:**
    - Perform data pre-processing steps such as handling missing values
    - Treating outliers
    - Applying feature scaling where necessary
    
    **2. Data Visualization:**
    - Use appropriate data visualization techniques (histograms, scatterplots, heatmaps, boxplots, etc.)
    - Explore the distribution of features and relationships between variables
    
    **3. Principal Component Analysis (PCA):**
    - Apply Principal Component Analysis (PCA) to reduce the dimensionality of the dataset
    - Retain maximum variance
    - Visualize the data in the transformed feature space
    - Interpret the results
    
    **4. Analysis & Insights:**
    - Discuss the patterns or clusters observed in the dataset
    - Explain how they could be useful for decision-making
    
    ---
    **Section:** ECE-I  
    **Team Members:** S.V. Chirudeep Reddy (64), U. Ganesh (57), Kyathendra Venkata Surya (11), Dev Rehanth (55)
    """)

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
        st.markdown('<style>body { background-color: #0E5517; color: white; }</style>', unsafe_allow_html=True)
    
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
        
        # Calculate correlation and handle NaN values
        corr = df[numeric_cols].corr()
        
        # Remove columns/rows that are all NaN
        corr_clean = corr.dropna(how='all', axis=0).dropna(how='all', axis=1)
        
        # Fill remaining NaN values with 0 (no correlation)
        corr_clean = corr_clean.fillna(0)
        
        # Create improved heatmap with no gaps
        fig3 = go.Figure(data=go.Heatmap(
            z=corr_clean.values,
            x=corr_clean.columns.tolist(),
            y=corr_clean.columns.tolist(),
            colorscale='RdBu_r',  # Reversed: Red=positive, Blue=negative
            zmid=0,  # Center the colorscale at 0
            text=np.round(corr_clean.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10, "color": "black"},
            colorbar=dict(title="Correlation"),
            hoverongaps=False,
            xgap=1,  # Small gap for better visibility
            ygap=1   # Small gap for better visibility
        ))
        
        fig3.update_layout(
            title={
                'text': "Feature Correlation Matrix",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#667eea', 'family': 'Arial Black'}
            },
            xaxis={
                'side': 'bottom',
                'tickangle': -45,
                'tickfont': {'size': 10}
            },
            yaxis={
                'autorange': 'reversed',
                'tickfont': {'size': 10}
            },
            width=None,  # Let it be responsive
            height=max(600, len(corr_clean) * 50),  # Dynamic height based on features
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig3, use_container_width=True)
        
        # Show explanation
        with st.expander("ℹ️ How to Read the Correlation Heatmap"):
            st.markdown("""
            - **Dark Red** (close to -1): Strong negative correlation
            - **White** (close to 0): No correlation
            - **Dark Blue** (close to 1): Strong positive correlation
            - **Diagonal = 1**: Each variable perfectly correlates with itself
            - Values removed: Columns with no variance or all missing data
            """)
    
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

# Footer with Team Information
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 30px; background: white; border: 2px solid black; border-radius: 10px; margin-top: 40px;'>
    <h3 style='color: black; margin-bottom: 15px;'> Meet the Team - ECE-I Section</h3>
    <div style='display: flex; justify-content: center; flex-wrap: wrap; gap: 20px; margin-top: 20px;'>
        <div style='background: black; padding: 15px 25px; border-radius: 8px; border: 1px solid black;'>
            <p style='margin: 0; color: white; font-weight: 600;'>S.V. Chirudeep Reddy</p>
            <p style='margin: 5px 0 0 0; color: white; font-size: 12px;'>Roll No: 64</p>
        </div>
        <div style='background: black; padding: 15px 25px; border-radius: 8px; border: 1px solid black;'>
            <p style='margin: 0; color: white; font-weight: 600;'>U. Ganesh</p>
            <p style='margin: 5px 0 0 0; color: white; font-size: 12px;'>Roll No: 57</p>
        </div>
        <div style='background: black; padding: 15px 25px; border-radius: 8px; border: 1px solid black;'>
            <p style='margin: 0; color: white; font-weight: 600;'>Kyathendra Venkata Surya</p>
            <p style='margin: 5px 0 0 0; color: white; font-size: 12px;'>Roll No: 11</p>
        </div>
        <div style='background: black; padding: 15px 25px; border-radius: 8px; border: 1px solid black;'>
            <p style='margin: 0; color: white; font-weight: 600;'>Dev Rehanth</p>
            <p style='margin: 5px 0 0 0; color: white; font-size: 12px;'>Roll No: 55</p>
        </div>
    </div>
    <p style='margin-top: 20px; color: black; font-size: 14px;'>
        Created with  for Data Science Assignment | © 2025
    </p>
</div>
""", unsafe_allow_html=True)


