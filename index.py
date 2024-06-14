import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_recall_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

churn_data = pd.read_csv('CSV/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Identificar e remover colunas irrelevantes
irrelevant_cols = ['customerID']
churn_data = churn_data.drop(columns=irrelevant_cols)

# Identificar e remover colunas duplicadas
churn_data = churn_data.loc[:, ~churn_data.columns.duplicated()]

# Dividir o conjunto de dados em variáveis independentes (X) e variável dependente (y)
X = churn_data.drop(columns=['Churn'])
y = churn_data['Churn']

# Dividir os dados em conjuntos de treinamento e teste 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir colunas categóricas e numéricas
categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(include=['number']).columns

# Pipeline para pré-processamento
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Tratar valores ausentes com a mediana
    ('scaler', StandardScaler())  # Padronizar variáveis numéricas
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Tratar valores ausentes com a constante 'missing'
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Codificar variáveis categóricas
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Pipeline completo
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000)) 
])

# Validação cruzada
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
print(f'Acurácia média da validação cruzada: {cv_scores.mean():.2f}')
print(f'Desvio padrão da acurácia da validação cruzada: {cv_scores.std():.2f}')

# Grid Search para otimização de hiperparâmetros
param_grid = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__solver': ['liblinear', 'saga']
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
print(f'Melhores parâmetros: {grid_search.best_params_}')
pipeline = grid_search.best_estimator_

# Treinar o modelo no conjunto de treinamento
pipeline.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = pipeline.predict(X_test)

# Codificar os rótulos verdadeiros e as previsões
label_encoder = LabelEncoder()
y_test_encoded = label_encoder.fit_transform(y_test)
y_pred_encoded = label_encoder.transform(y_pred)

# Avaliar o desempenho do modelo com os rótulos codificados
accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
precision = precision_score(y_test_encoded, y_pred_encoded, pos_label=1) 
recall = recall_score(y_test_encoded, y_pred_encoded, pos_label=1) 
f1 = f1_score(y_test_encoded, y_pred_encoded, pos_label=1) 

print("Acurácia:", accuracy)
print("Precisão:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Relatório de Classificação
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

# Curva ROC e AUC
y_prob = pipeline.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test_encoded, y_prob)
roc_auc = roc_auc_score(y_test_encoded, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Matriz de Confusão
conf_matrix = confusion_matrix(y_test, y_pred, labels=label_encoder.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_encoder.classes_)
disp.plot(cmap='Blues')
plt.title('Matriz de Confusão')
plt.show()

# Curva de Precisão-Recall
precision_vals, recall_vals, _ = precision_recall_curve(y_test_encoded, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, label='Curva de Precisão-Recall')
plt.xlabel('Recall')
plt.ylabel('Precisão')
plt.title('Curva de Precisão-Recall')
plt.legend(loc='lower left')
plt.grid(True)
plt.show()

# Análise de Resíduos
y_prob_train = pipeline.predict_proba(X_train)[:, 1]
y_train_encoded = label_encoder.transform(y_train)  
residuals = y_train_encoded - y_prob_train

plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=50)
plt.xlabel('Resíduos Deviance')
plt.ylabel('Frequência')
plt.title('Histograma dos Resíduos')
plt.grid(True)
plt.show()

# Importância das Variáveis (coeficientes)
model = pipeline.named_steps['classifier']
coefs = model.coef_[0]

cat_features = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_cols)
features = np.concatenate([numeric_cols, cat_features])

coef_df = pd.DataFrame({'Feature': features, 'Coefficient': coefs})
coef_df = coef_df.sort_values(by='Coefficient', ascending=False)

# Importâncias das variáveis numéricas
numeric_coef_df = coef_df[coef_df['Feature'].isin(numeric_cols)]
plt.figure(figsize=(10, 6))
plt.barh(numeric_coef_df['Feature'], numeric_coef_df['Coefficient'], color='lightblue')
plt.xlabel('Importância')
plt.ylabel('Feature')
plt.title('Importância das Variáveis Numéricas')
plt.grid(True)
plt.show()
