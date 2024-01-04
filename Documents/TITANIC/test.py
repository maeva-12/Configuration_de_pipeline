import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

# Charger le modèle
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Charger ou préparer les données de test
df = pd.read_csv('sample.csv',sep=';')
X_test = df.drop('target', axis=1)
y_test = df['target']

# Faire des prédictions
y_pred = model.predict(X_test)

# Calculer la précision
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle : {accuracy:.2f}")

# Définir un seuil de classification
seuil_classification = 0.90  # par exemple

# Vérifier si le seuil est atteint
if accuracy >= seuil_classification:
    print("Le seuil de classification est atteint. Le modèle est prêt pour le déploiement.")
else:
    print("Le seuil de classification n'est pas atteint. Le modèle nécessite une amélioration.")