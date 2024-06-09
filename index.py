import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

## Explorando e Preparando os Dados

churn_data = pd.read_csv("CSV/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Visualizar as primeiras linhas do conjunto de dados
print(churn_data.head())

# Obter informações sobre as características e variáveis do conjunto de dados
print(churn_data.info())

# Estatísticas descritivas das variáveis numéricas
print(churn_data.describe())

# Verificar se há valores ausentes
print(churn_data.isnull().sum())

## Desenvolvendo o modelo de regressão

X = churn_data.drop(columns=['Churn'])
y = churn_data['Churn']

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir colunas categóricas e numéricas
categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(include=['number']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Pipeline completo
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))  # Aumentando o max_iter
])

# Treinar o modelo no conjunto de treinamento
pipeline.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = pipeline.predict(X_test)

# Avaliar o desempenho do modelo
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='Yes')  # Especificando 'Yes' como rótulo positivo
recall = recall_score(y_test, y_pred, pos_label='Yes')  # Especificando 'Yes' como rótulo positivo
f1 = f1_score(y_test, y_pred, pos_label='Yes')  # Especificando 'Yes' como rótulo positivo

print("Acurácia:", accuracy)
print("Precisão:", precision)
print("Recall:", recall)
print("F1-score:", f1)

## Interpretando e analisando os resultados

# Obter os coeficientes do modelo
coeficients = pipeline.named_steps['classifier'].coef_[0]

numeric_features = list(numeric_cols)
categorical_features = pipeline.named_steps['preprocessor'].named_transformers_['cat'] \
                            .get_feature_names_out(input_features=categorical_cols)
feature_names = numeric_features + list(categorical_features)

# Obter os índices ordenados dos coeficientes
sorted_indices = np.argsort(coeficients)

# Ordenar os coeficientes e os nomes das variáveis de acordo com os índices
sorted_coeficients = coeficients[sorted_indices]
sorted_feature_names = np.array(feature_names)[sorted_indices]

plt.figure(figsize=(10, 6))
plt.barh(sorted_feature_names, sorted_coeficients)
plt.xlabel('Coeficiente')
plt.title('Coeficientes do Modelo de Regressão Logística')
plt.grid(True)
plt.gca().axes.get_yaxis().set_visible(False)  # Remover marcação do eixo y
plt.show()


# Matriz de confusão

# Calcular a matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Previsão')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão')
plt.show()

