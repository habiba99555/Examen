import streamlit as st
import pandas as pd
import numpy as np
#from sklearn.preprocessing import StandardScaler, LabelEncoder

# Fonction de chargement du fichier
@st.cache_data
def load_data(file):
    if file is not None:
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.endswith('.xlsx'):
                df = pd.read_excel(file)
            elif file.name.endswith('.json'):
                df = pd.read_json(file)
            else:
                st.error("Format non supporté. Utilisez CSV, Excel ou JSON.")
                return None
            return df
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier : {e}")
            return None
    return None

# Fonction de nettoyage des données
def clean_data(df, missing_threshold=0.6):
    if df is None:
        return None

    df_cleaned = df.copy()

    # Suppression des colonnes avec trop de valeurs manquantes
    df_cleaned = df_cleaned.dropna(thresh=int(missing_threshold * len(df_cleaned)), axis=1)

    # Remplacement des valeurs manquantes par la moyenne pour les colonnes numériques
    df_cleaned.fillna(df_cleaned.mean(numeric_only=True), inplace=True)

    # Encodage des variables catégorielles
    label_encoders = {}
    for col in df_cleaned.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_cleaned[col] = le.fit_transform(df_cleaned[col].astype(str))
        label_encoders[col] = le

    # Normalisation des valeurs numériques
    scaler = StandardScaler()
    numeric_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
    df_cleaned[numeric_cols] = scaler.fit_transform(df_cleaned[numeric_cols])

    return df_cleaned

# Interface Streamlit
st.title("🧮 Prétraitement du MathE Dataset")
st.write("Téléchargez un fichier de données et effectuez son prétraitement.")

# Upload du fichier
uploaded_file = st.file_uploader("📂 Importer un fichier (CSV, Excel, JSON)", type=["csv", "xlsx", "json"])

if uploaded_file is not None:
    # Charger et afficher les données
    df = load_data(uploaded_file)
    
    if df is not None:
        st.subheader("🔍 Aperçu des données originales")
        st.dataframe(df.head())

        # Affichage des informations générales
        st.write("**📊 Informations générales sur le dataset :**")
        buffer = df.info(buf=None)
        st.text(buffer)

        # Affichage des valeurs manquantes
        st.write("**❗ Valeurs manquantes par colonne :**")
        missing_values = df.isnull().sum()
        st.write(missing_values[missing_values > 0])

        # Nettoyage des données
        st.subheader("⚙️ Prétraitement des données")
        cleaned_df = clean_data(df)

        if cleaned_df is not None:
            st.write("✅ Données nettoyées et transformées")
            st.dataframe(cleaned_df.head())

            # Télécharger le fichier nettoyé
            csv = cleaned_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="⬇️ Télécharger les données prétraitées",
                               data=csv,
                               file_name="MathE_dataset_cleaned.csv",
                               mime="text/csv")

