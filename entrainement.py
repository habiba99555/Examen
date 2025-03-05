import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# 1. Charger les données prétraitées
df = pd.read_csv("MathE_dataset_cleaned.csv")  # Remplace par le bon fichier

# 2. Séparer les caractéristiques (X) et la cible (y)
target_column = "target"  # Remplace par la colonne à prédire
if target_column not in df.columns:
    raise ValueError(f"La colonne cible '{target_column}' n'existe pas dans le dataset.")

X = df.drop(columns=[target_column])
y = df[target_column]

# Encodage de la variable cible si elle est catégorielle
if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)

# 3. Séparer les données en ensemble d'entraînement et de test (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialiser et entraîner le modèle (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Évaluer le modèle
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Précision du modèle : {accuracy:.2f}")
print("\n🔍 Rapport de classification :")
print(classification_report(y_test, y_pred))

# 6. Sauvegarder le modèle pour une utilisation future
joblib.dump(model, "modele_MathE.pkl")
print("✅ Modèle sauvegardé sous 'modele_MathE.pkl'.")

