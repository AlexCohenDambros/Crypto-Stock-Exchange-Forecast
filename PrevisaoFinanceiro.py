# Importações
# Modelo usado -> LSTM
from turtle import end_fill
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM,Dropout

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas_datareader import data as web


dataframe_dados = web.DataReader('BTC-USD', data_source="yahoo", start="01-01-2018", end="02-22-2022")

dataframe_dados_selecionados = dataframe_dados[['Adj Close']]
dataframe_dados_selecionados = dataframe_dados_selecionados.set_index(dataframe_dados.index)

# plotar informações
plt.figure(figsize=(16,8))
plt.title('Preço de fechamento')
plt.plot(dataframe_dados_selecionados['Adj Close'])
plt.xlabel('data',fontsize=18)
plt.yticks(np.arange(0, max(dataframe_dados_selecionados['Adj Close']), 5000))
plt.show()

# quantidade de linhas 
quantidade_linhas = len(dataframe_dados_selecionados)
quantidade_linhas_treino = round(.70 * quantidade_linhas)
quantidade_linhas_teste = quantidade_linhas - quantidade_linhas_treino
info = ( 
    f"linhas treino = 0:{quantidade_linhas_treino}"
    f" linhas teste = {quantidade_linhas_treino}:{quantidade_linhas_treino + quantidade_linhas_teste}"
)

print(info)

# normalização dos dados
scaler = StandardScaler()
df_scaled = scaler.fit_transform(dataframe_dados_selecionados)

# separar em treino e teste
treino = df_scaled[:quantidade_linhas_treino]
teste = df_scaled[quantidade_linhas_treino: quantidade_linhas_treino + quantidade_linhas_teste]

# converter um array de valores em um dataframe matriz
def converte_array(dataframe, steps=1):
  dataX, dataY = [], []
  for x in range(len(dataframe)-steps-1):
    var = dataframe[x:(x+steps), 0]
    dataX.append(var)
    dataY.append(dataframe[x + steps, 0])
  return np.array(dataX), np.array(dataY)

# gerando dados de treino e teste
steps = 10
x_treino, y_treino = converte_array(treino, steps)
x_teste, y_teste = converte_array(teste, steps)

#print(x_treino.shape)
#print(y_treino.shape)
#print(x_teste.shape)
#print(y_teste.shape)

# gerando os dados que o modelo espera
x_treino = x_treino.reshape(x_treino.shape[0], x_treino.shape[1], 1)
x_teste = x_teste.reshape(x_teste.shape[0], x_teste.shape[1], 1)

# montando a rede neural
modelo = Sequential()
modelo.add(LSTM(35, return_sequences=True, input_shape=(steps, 1)))
modelo.add(LSTM(35, return_sequences=True))
modelo.add(LSTM(35))
modelo.add(Dropout(0.2))
modelo.add(Dense(1)) # saida

modelo.compile(optimizer="adam", loss="mse")
#modelo.summary()

# treinamento do modelo 
validacao = modelo.fit(x_treino, y_treino, validation_data=(x_teste, y_teste), epochs=100 , batch_size=10 , verbose=2)

# visualizar do treinamento
plt.plot(validacao.history['loss'], label='Perda de treinamento')
plt.plot(validacao.history['val_loss'], label='Perda de validação')
plt.legend()

# Fazendo a Previsão
previsao = modelo.predict(x_teste)
previsao = scaler.inverse_transform(previsao)
#print(previsao)

# previsão para os proximos 10 dias 
lenght_teste = len(teste)
#print(lenght_teste)

# pegar os ultimos dias que sao o tamanho do meu step
dias_input_steps = lenght_teste - steps
#print(dias_input_steps)

# transformando em array
input_steps = teste[dias_input_steps:]
input_steps = np.array(input_steps).reshape(1, -1)
#print(input_steps)

# transformar em lista 
lista_output_steps = list(input_steps)
lista_output_steps = lista_output_steps[0].tolist()
#print(lista_output_steps)

# loop para prever os proximos 10 dias
lista_dias = []
x = 0
numero_dias = 10

while(x < numero_dias):
  if (len(lista_output_steps) > steps):
    input_steps = np.array(lista_output_steps[1:])

    #print("{} dia. Valores de entrada -> {}".format(x, input_steps))

    input_steps = input_steps.reshape(1,-1)
    input_steps = input_steps.reshape((1, steps, 1))

    previsao = modelo.predict(input_steps, verbose = 0)
    #print("{} dia. Valor previsto - > {}".format(x, previsao))

    lista_output_steps.extend(previsao[0].tolist())
    lista_output_steps = lista_output_steps[1:]

    lista_dias.extend(previsao.tolist())
     
    x += 1
  else:
    input_steps = input_steps.reshape((1, steps, 1))
    previsao = modelo.predict(input_steps, verbose=0)
    
    #print(previsao[0])

    lista_output_steps.extend(previsao[0].tolist())

    #print(len(lista_output_steps))

    lista_dias.extend(previsao.tolist())

    x += 1

#print(lista_dias)

# transformando a saida
previsao = scaler.inverse_transform(lista_dias)
previsao = np.array(previsao).reshape(1 , -1)
lista_output_prev = list(previsao)
lista_output_prev = previsao[0].tolist()
#print(lista_output_prev)

# pegando as datas de previsao
datas = pd.to_datetime(dataframe_dados.index)
predict_dates = pd.date_range(list(datas)[-1] + pd.DateOffset(1), periods = 10, freq="b").tolist()
#print(predict_dates)

# criar dataframe de previsão
forecast_dates = []

for i in predict_dates:
  forecast_dates.append(i.date())

df_forecast = pd.DataFrame({'Date': np.array(forecast_dates), 'Valor Previsto': lista_output_prev})
df_forecast["Date"] = pd.to_datetime(df_forecast["Date"])

df_forecast = df_forecast.set_index(pd.DatetimeIndex(df_forecast["Date"]).values)
df_forecast.drop('Date', axis=1, inplace=True)
print(df_forecast)

# plotando o gráfico
plt.figure(figsize=(16,8))
plt.plot(dataframe_dados_selecionados["Adj Close"])
plt.plot(df_forecast["Valor Previsto"])
plt.legend(["Preço de Fechamento", "Preço Previsto"])
plt.show()
