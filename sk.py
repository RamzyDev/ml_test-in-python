import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("patients.csv")
print(data.info())
print(data.isnull().sum())
print(data.head())

def preprocess_data(df):
    df = df.copy()
    df['sexe'] = df['sexe'].map({'H': 0, 'F': 1})
    df = pd.get_dummies(df, columns=['diagnostic'], drop_first=True)
    df.drop(['id', 'nom'], axis=1, inplace=True)
    return df

data_clean = preprocess_data(data)

if 'diagnostic_Hypertension' in data_clean.columns:
    X = data_clean.drop('diagnostic_Hypertension', axis=1)
    y = data_clean['diagnostic_Hypertension']
else:
    print(" Aucune colonne 'diagnostic_Hypertension' à prédire.")
    X = data_clean
    y = np.zeros(len(data_clean))

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

if len(X) >= 3:
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    param_grid = {'n_neighbors': list(range(1, min(6, len(X_train) + 1)))}
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=2)
    grid.fit(X_train, y_train)

    def evaluate_model(model, X_test, y_test):
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        print(f"Accuracy: {acc:.2f}")
        sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm')
        plt.title("Matrice de confusion")
        plt.xlabel("Prédit")
        plt.ylabel("Réel")
        plt.show()

    evaluate_model(grid.best_estimator_, X_test, y_test)

else:
    print(" Pas assez de données pour entraîner un modèle (au moins 3 lignes nécessaires).")

isolation_forest = IsolationForest(contamination=0.1, random_state=42)
anomaly_labels = isolation_forest.fit_predict(X_scaled)

normal_data = X_scaled[anomaly_labels == 1]
anomalies = X_scaled[anomaly_labels == -1]
plt.figure(figsize=(10, 6))
plt.scatter(normal_data[:, 0], normal_data[:, 1], color='darkgreen', label='Normal', alpha=0.7)
plt.scatter(anomalies[:, 0], anomalies[:, 1], color='darkorange', label='Anomalie', alpha=0.7)
plt.title("Détection d'anomalies avec Isolation Forest")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
