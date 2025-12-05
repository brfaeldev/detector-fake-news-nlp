import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def plotar_matriz():
    # Caminhos
    caminho_dados = os.path.join("data", "processed", "dados_limpos.csv")
    caminho_modelo = os.path.join("data", "models", "modelo_fake_news.pkl")
    caminho_vetor = os.path.join("data", "models", "vetorizador.pkl")
    
    print("Carregando dados e modelo...")
    df = pd.read_csv(caminho_dados).dropna()
    modelo = joblib.load(caminho_modelo)
    vetorizador = joblib.load(caminho_vetor)
    
    # Vetoriza tudo para testar
    X = vetorizador.transform(df['texto_final'])
    y_real = df['label']
    
    # Previsões
    print("Gerando previsões para o gráfico...")
    y_pred = modelo.predict(X)
    
    # Gera a matriz
    cm = confusion_matrix(y_real, y_pred)
    
    # Configura o gráfico
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Previsto: Real', 'Previsto: Fake'],
                yticklabels=['É Real', 'É Fake'])
    
    plt.title('Matriz de Confusão - Performance da IA')
    plt.ylabel('Verdade (Dataset)')
    plt.xlabel('O que a IA disse')
    
    # Salva
    caminho_img = os.path.join("data", "matriz_confusao.png")
    plt.savefig(caminho_img)
    print(f"Gráfico salvo em: {caminho_img}")
    print("Abra essa imagem para ver o resultado visual!")

if __name__ == "__main__":
    plotar_matriz()
    