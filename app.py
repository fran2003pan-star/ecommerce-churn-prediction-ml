import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Para el modelo y evaluación
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')


# 1. Definir la ruta del archivo
file_path = 'customer_churn_dataset-training-master.csv'




# 2. Cargar el dataset
try:
    df = pd.read_csv(file_path)
    print("✅ ¡Dataset cargado con éxito!")
    
    # Ver dimensiones
    print(f"\n📊 Dimensiones: {df.shape[0]} filas y {df.shape[1]} columnas")
    
    # Ver tipos de datos y nulos
    print("\n🧐 Información del Dataset:")
    print(df.info())
    
    # Ver cuántos clientes se van vs cuántos se quedan (Desbalanceo)
    if 'Churn' in df.columns:
        print("\n⚖️ Distribución de la variable objetivo (Churn):")
        print(df['Churn'].value_counts(normalize=True) * 100)
    
except FileNotFoundError:
    print("❌ Error: Asegúrate de que el CSV esté en la misma carpeta que app.py")




    # 3. Visualización: Impacto de las llamadas a soporte en la fuga
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='Support Calls', hue='Churn', fill=True, palette='viridis')
plt.title('Distribución de Llamadas a Soporte: Clientes Activos vs Fugados')
plt.xlabel('Número de Llamadas a Soporte')
plt.ylabel('Densidad')
plt.grid(axis='y', alpha=0.3)

# Guardar la imagen para el portfolio
plt.savefig('grafica_soporte_churn.png')
print("\n✅ Gráfica generada y guardada como 'grafica_soporte_churn.png'")

# Mostrar la gráfica
plt.show()




print("\n--- Paso 3: Preprocesamiento de Datos (Encoding) ---")

# 4. Crear una copia para no estropear el df original
df_encoded = df.copy()

# 5. Borrar columnas inútiles para el modelo (como el ID)
if 'CustomerID' in df_encoded.columns:
    df_encoded = df_encoded.drop(columns=['CustomerID'])
    print("✅ Columna 'CustomerID' eliminada.")

# 6. Convertir texto a números (One-Hot Encoding)
# Usamos One-Hot porque estas categorías no tienen un orden lógico (ej: Premium no es mayor que Basic, son distintos).
df_encoded = pd.get_dummies(df_encoded, columns=['Gender', 'Subscription Type', 'Contract Length'], drop_first=True)

print("\n✅ Encoding completado con éxito.")
print(f"Dimensiones después del encoding: {df_encoded.shape[0]} filas y {df_encoded.shape[1]} columnas")
print("\nPrimeras 5 filas con datos numéricos:")
print(df_encoded.head())
# --- LIMPIEZA DE ÚLTIMO MOMENTO ---
# Eliminar cualquier fila que tenga valores nulos (NaN) que hayan pasado desapercibidos
df_encoded = df_encoded.dropna()
print(f"✅ Limpieza final: {df_encoded.shape[0]} filas listas para el modelo.")

# --- Paso 4: Entrenamiento del Modelo (Random Forest) ---
X = df_encoded.drop(columns=['Churn'])
y = df_encoded['Churn']

# El resto del código sigue igual...
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# ...



print("\n--- Paso 4: Entrenamiento del Modelo (Random Forest) ---")

# 1. Separar las variables (X) de lo que queremos predecir (y)
X = df_encoded.drop(columns=['Churn'])
y = df_encoded['Churn']

# 2. Dividir en set de Entrenamiento (80%) y Prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Crear y entrenar el modelo
print("🚀 Entrenando el modelo (esto puede tardar unos segundos debido al tamaño del dataset)...")
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# 4. Realizar predicciones
y_pred = model.predict(X_test)

# 5. Evaluación: Matriz de Confusión Visual
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matriz de Confusión: Predicción de Churn')
plt.xlabel('Predicción del Modelo')
plt.ylabel('Realidad (Valor Real)')
plt.savefig('matriz_confusion.png')
print("\n✅ Matriz de confusión guardada como 'matriz_confusion.png'")

# 6. Informe de métricas
print("\n📈 Reporte de Clasificación:")
print(classification_report(y_test, y_pred))

plt.show()



# --- Paso 5: Importancia de las Variables ---
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Gráfica de barras
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='magma')
plt.title('Factores Clave en la Fuga de Clientes (Feature Importance)')
plt.xlabel('Importancia Relativa')
plt.ylabel('Variable')
plt.tight_layout()

# Guardar la última imagen
plt.savefig('importancia_variables.png')
print("\n✅ Gráfica de importancia guardada como 'importancia_variables.png'")

plt.show()