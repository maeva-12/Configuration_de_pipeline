import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

# Charger le modèle
with open('breast_cancer_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Charger ou préparer les données de test
url = 'https://raw.githubusercontent.com/AbdallahTayeb/DevOps-Course/main/sample.csv'
test_data = pd.read_csv(url)

# Supprimer la colonne cible (si elle existe)
if 'target' in test_data.columns:
    test_data.drop('target', axis=1, inplace=True)

# Faire des prédictions
y_pred = model.predict(test_data)

# Calculer la précision
# Note : Vous aurez besoin de véritables valeurs cibles pour calculer la précision
# Si le fichier sample.csv contient également la colonne target, vous pouvez la comparer avec les prédictions
# Sinon, vous devrez ajuster cette partie en fonction de vos besoins.

# Charger les vraies valeurs cibles depuis le fichier sample.csv (si disponible)
true_values = test_data['target']  # Assurez-vous que la colonne target existe dans le fichier sample.csv

# Calculer la précision
accuracy = accuracy_score(true_values, y_pred)
print(f"Précision du modèle sur les nouvelles données : {accuracy:.2f}")

# Définir un seuil de classification
seuil = 0.8

# Vérifier si le seuil est atteint
predictions_at_seuil = (model.predict_proba(test_data)[:, 1] > seuil).astype(int)
print(f"Précision avec seuil {seuil} : {accuracy_score(true_values, predictions_at_seuil):.2f}")
