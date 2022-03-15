# Web App Para Previsões em Tempo Real com Machine Learning em Python

# Imports
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Título
st.header("Web App Para Previsões em Tempo Real com Machine Learning em Python")

# Sub-título
st.subheader('Prevendo a Ocorrência de Diabetes em Novos Pacientes')

# Texto
st.write('Tabela com os dados originais:')

# Carrega os dados
df = pd.read_csv('diabetes.csv')

# Imprime o dataframe
st.dataframe(df)

# Texto
st.write('Tabela com resumo estatístico dos dados:')

# Resumo estatístico
st.write(df.describe())

# Texto
st.write('Visualizando os dados originais com gráfico de barras:')

# Visualizando os dados
chart = st.bar_chart(df)

# Divisão dos dados em entrada (X) e saída (Y)
# Y é o que queremos prever, nesse caso a ocorrência de doença
X = df.iloc[:, 0:8].values
Y = df.iloc[:, -1].values

# Divisão dos dados em treino e teste com proporção 75/25
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# Função para obter input do usuário (novos dados)
def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('glucose', 0, 199, 117)
    blood_pressure = st.sidebar.slider('blood_pressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('skin_thickness', 0, 99, 23)
    insulin = st.sidebar.slider('insulin', 0.0, 846.0, 30.0)
    bmi = st.sidebar.slider('bmi', 0.0, 67.1, 32.0)
    dpf = st.sidebar.slider('dpf', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('age', 21, 81, 29)

    # Cria um dicionário com os dados
    user_data = { 'pregnancies' : pregnancies,
                  'glucose' : glucose,
                  'blood_pressure' : blood_pressure,
                  'skin_thickness' : skin_thickness,
                  'insulin' : insulin,
                  'bmi' : bmi,
                  'dpf' : dpf,
                  'age' : age
               }

    # Transforma os dados em um dataframe
    features = pd.DataFrame(user_data, index = [0])

    return features

# Armazena o input do usuário
user_input = get_user_input()

# Sub-título
st.subheader('Input do usuário (novos dados) : ')
st.write(user_input)

# Cria o modelo de ML
modelo = RandomForestClassifier()

# Treina o modelo
modelo.fit(X_treino, Y_treino)

# Imprime a acurácia do modelo
st.subheader("Acurácia do Modelo : " + str(accuracy_score(Y_teste, modelo.predict(X_teste))* 100 ) + '%')

# Faz as previsões
prediction = modelo.predict(user_input.values)

# Imprime o resultado
if prediction[0] == 1:
  st.subheader('Esse paciente, provavelmente, deve desenvolver diabetes!')
else:
  st.subheader('Esse paciente, provavelmente, não deve desenvolver diabetes!')




