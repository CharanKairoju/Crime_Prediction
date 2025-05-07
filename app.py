import pandas as pd # type: ignore
import streamlit as st # type: ignore
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.cluster import KMeans # type: ignore

st.set_page_config(page_title="Crime Prediction System", layout="wide")
st.title("🕵️ Crime Prediction and Prevention Dashboard")

uploaded_file = st.file_uploader("Upload Crime Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data")
    st.dataframe(df.head())

    st.subheader("Crime Statistics")
    st.write("Total crimes:", len(df))
    st.write("First few rows of your DataFrame:")
    st.write(df.head())
    st.write("Column names in your DataFrame:")
    st.write(df.columns)
    st.write("Crime Types:", df['Crime Type'].value_counts().head())

    st.subheader("Heatmap of High-Crime Locations")
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        fig, ax = plt.subplots()
        sns.kdeplot(x=df["Longitude"], y=df["Latitude"], cmap="Reds", fill=True, ax=ax)
        st.pyplot(fig)

    st.subheader("Predict Crime Hotspots (KMeans Clustering)")
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        kmeans = KMeans(n_clusters=5)
        df['Cluster'] = kmeans.fit_predict(df[['Latitude', 'Longitude']])
        st.map(df[['Latitude', 'Longitude']])

        st.write("Cluster Centers:")
        st.write(pd.DataFrame(kmeans.cluster_centers_, columns=["Latitude", "Longitude"]))
