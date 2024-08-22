import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import requests

# Carregar os dados de treinamento do arquivo CSV
df_train = pd.read_csv('diabetes_dataset.csv')

# Carregar os dados de inferência do arquivo CSV
df_infer = pd.read_csv('diabetes_app.csv')

# Preencher dados categóricos nulos com um valor específico, como 'Desconhecido'
df_train = df_train.fillna(0)

# Separar features e target dos dados de treinamento
X_train = df_train.iloc[:, :-1]  # Todas as colunas, exceto a última
y_train = df_train.iloc[:, -1]   # Apenas a última coluna

# Separar features dos dados de inferência
X_infer = df_infer

# Criar o modelo KNN com k=3 (por exemplo)
knn = KNeighborsClassifier(n_neighbors=3)

# Treinar o modelo
knn.fit(X_train, y_train)

# Fazer previsões nos dados de inferência
y_pred = knn.predict(X_infer)

# Exibir os resultados
print(y_pred)

# Enviando previsões realizadas com o modelo para o servidor
URL = "https://aydanomachado.com/mlclass/01_Preprocessing.php"

#TODO Substituir pela sua chave aqui
DEV_KEY = "Mexerica"

# json para ser enviado para o servidor
data = {'dev_key':DEV_KEY,
        'predictions':pd.Series(y_pred).to_json(orient='values')}

# Enviando requisição e salvando o objeto resposta
r = requests.post(url = URL, data = data)

# Extraindo e imprimindo o texto da resposta
pastebin_url = r.text
print(" - Resposta do servidor:\n", r.text, "\n")