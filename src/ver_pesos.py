import joblib
import pandas as pd
import os

def ver_pesos():
    print("Carregando modelo...")
    caminho_modelo = os.path.join("data", "models", "modelo_fake_news.pkl")
    caminho_vetor = os.path.join("data", "models", "vetorizador.pkl")
    
    # Carrega a IA treinada
    modelo = joblib.load(caminho_modelo)
    vetorizador = joblib.load(caminho_vetor)
    
    # Pega todas as palavras que a IA aprendeu
    palavras = vectorizer_names = vetorizador.get_feature_names_out()
    
    # Pega os pesos (importância) de cada palavra
    # Pesos positivos - puxam para FAKE (1)
    # Pesos negativos - puxam para REAL (0)
    pesos = modelo.coef_[0]
    
    # Cria uma tabela
    df = pd.DataFrame({'palavra': palavras, 'peso': pesos})
    
    # As 20 palavras que mais indicam que é FAKE
    top_fake = df.sort_values(by='peso', ascending=False).head(20)
    
    # As 20 palavras que mais indicam que é REAL
    top_real = df.sort_values(by='peso', ascending=True).head(20)
    
    print("\n==============================================")
    print(" TOP 20 PALAVRAS -> IA DIZ QUE É FAKE (1)")
    print("==============================================")
    print(top_fake)
    
    print("\n==============================================")
    print(" TOP 20 PALAVRAS -> IA DIZ QUE É REAL (0)")
    print("==============================================")
    print(top_real)
    
    # Checagem do "Intercepto" - (o viés padrão da IA quando não conhece as palavras)
    print("\n==============================================")
    print(f" VIÉS PADRÃO (Intercept): {modelo.intercept_[0]:.4f}")
    print(" (Se for positivo, na dúvida ela chuta Fake. Se negativo, chuta Real)")
    print("==============================================")

if __name__ == "__main__":
    ver_pesos()