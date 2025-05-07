import pandas as pd  # type: ignore
import streamlit as st  # type: ignore
import seaborn as sns  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from sklearn.cluster import KMeans  # type: ignore

st.set_page_config(page_title="Crime Prediction System", layout="wide")
st.title("🕵️ Crime Prediction and Prevention Dashboard")

uploaded_file = st.file_uploader("Upload Crime Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Debugging: Display uploaded dataset columns
        st.write("Uploaded Dataset Columns (before removing duplicates):", df.columns.tolist())

        # Remove duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]
        st.write("Columns after removing duplicates:", df.columns.tolist())

        # Rename columns to match the required names
        column_mapping = {
            'Murder': 'Primary Type',  # Example mapping
            'LatitudeColumn': 'Latitude',  # Replace with the actual column name for latitude
            'LongitudeColumn': 'Longitude'  # Replace with the actual column name for longitude
        }
        df.rename(columns=column_mapping, inplace=True)

        # Check for required columns
        required_columns = ['Primary Type', 'Latitude', 'Longitude']
        if not all(col in df.columns for col in required_columns):
            st.error(
                f"The dataset must contain the following columns: {', '.join(required_columns)}. "
                f"Uploaded columns: {', '.join(df.columns)}"
            )
        else:
            # Data Cleaning: Drop rows with missing values in required columns
            df = df.dropna(subset=required_columns)

            st.subheader("Raw Data")
            st.dataframe(df.head())
            st.write("First few rows of your DataFrame:")
            st.write(df.head())
            st.write("Column names in your DataFrame:")
            st.write(df.columns)

            if 'Primary Type' in df.columns:
                st.subheader("Crime Statistics")
                st.write("Total crimes:", len(df))
                st.write("Crime Types:", df['Primary Type'].value_counts().head())
            else:
                st.error("The dataset does not contain the 'Primary Type' column.")

            if 'Latitude' in df.columns and 'Longitude' in df.columns:
                # Perform heatmap and clustering
                st.subheader("Heatmap of High-Crime Locations")
                fig, ax = plt.subplots()
                sns.kdeplot(
                    x=df["Longitude"], y=df["Latitude"], cmap="Reds", fill=True, ax=ax
                )
                ax.set_title("Heatmap of High-Crime Locations")
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
                st.pyplot(fig)

                st.subheader("Predict Crime Hotspots (KMeans Clustering)")
                cluster_count = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=5)
                kmeans = KMeans(n_clusters=cluster_count, random_state=42)
                df['Cluster'] = kmeans.fit_predict(df[['Latitude', 'Longitude']])

                # Map visualization with clusters
                st.map(df[['Latitude', 'Longitude']])

                st.write("Cluster Centers:")
                st.write(pd.DataFrame(kmeans.cluster_centers_, columns=["Latitude", "Longitude"]))
            else:
                st.error("The dataset does not contain 'Latitude' and 'Longitude' columns. Heatmap and clustering cannot be performed.")

            # Check for alternative columns if 'Primary Type' is not available
            if 'Murder' in df.columns and 'Rape' in df.columns:
                st.write("Using alternative crime columns: Murder, Rape, etc.")
                # Perform analysis based on these columns
            else:
                st.error("The dataset does not contain the required columns or alternatives.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a dataset with the following columns: 'Primary Type', 'Latitude', 'Longitude'.")
