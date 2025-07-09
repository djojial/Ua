import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
#  Sidebar Navigation
# ======================
st.sidebar.title("Main Page")
page = st.sidebar.radio(
    "Navigation",
    ["Classification", "Clustering"]
)

# ======================
#  Main Title
# ======================
st.title("Ujian Akhir Semester")
st.header("Streamlit Apps")
st.write("Collection of my apps deployed in Streamlit")
st.write("Nama: [ISI NAMAMU]")
st.write("NIM: [ISI NIMMU]")

# ======================
#  Classification Page
# ======================
if page == "Classification":
    st.subheader("Klasifikasi Diabetes")
    st.write("**Deskripsi Proyek**: Klasifikasi pasien berdasarkan kemungkinan diabetes menggunakan dataset Pima Indians Diabetes.")

    # Load data
    data = pd.read_csv("diabetes.csv")
    st.write("Sample Data:", data.head())

    # Split data
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Metrics
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    st.write("**Metrik Klasifikasi**")
    st.text(classification_report(y_test, y_pred))

    st.write("**Confusion Matrix**")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    st.write("**Input Data Baru**")
    input_data = {}
    for col in X.columns:
        val = st.number_input(f"Input {col}", value=float(X[col].mean()))
        input_data[col] = val

    input_df = pd.DataFrame([input_data])

    if st.button("Prediksi"):
        prediction = model.predict(input_df)
        st.write(f"**Hasil Prediksi:** {'Positif Diabetes' if prediction[0] == 1 else 'Negatif Diabetes'}")

# ======================
#  Clustering Page
# ======================
elif page == "Clustering":
    st.subheader("Clustering Lokasi Gerai Kopi")
    st.write("**Deskripsi Proyek**: Segmentasi lokasi gerai kopi menggunakan algoritma K-Means untuk menentukan cluster optimal.")

    # Load data
    data = pd.read_csv("lokasi_gerai_kopi_clean.csv")
    st.write("Sample Data:", data.head())

    # Only use numeric cols for clustering
    X = data.select_dtypes(include=['float64', 'int64'])

    # Run KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    data['Cluster'] = kmeans.fit_predict(X)

    st.write("**Visualisasi Hasil Clustering**")
    fig, ax = plt.subplots()
    scatter = ax.scatter(data.iloc[:, 0], data.iloc[:, 1], c=data['Cluster'], cmap='viridis')
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[1])
    plt.title("Hasil Clustering Lokasi Gerai Kopi")
    st.pyplot(fig)

    st.write("**Input Data Baru**")
    input_data = []
    for col in X.columns:
        val = st.number_input(f"Input {col}", value=float(X[col].mean()))
        input_data.append(val)

    if st.button("Prediksi Cluster"):
        cluster = kmeans.predict([input_data])
        st.write(f"**Hasil Clustering:** Cluster {cluster[0]}")

