import joblib
import re
import string
import os
import numpy as np

def limpar_texto(texto):
    if not isinstance(texto, str): return ""
    texto = texto.lower()
    texto = re.sub(r'https?://\S+|www\.\S+', '', texto)
    texto = re.sub(r'<.*?>', '', texto)
    texto = re.sub(r'[%s]' % re.escape(string.punctuation), '', texto)
    texto = re.sub(r'\n', ' ', texto)
    texto = re.sub(r'\w*\d\w*', '', texto)
    return texto

def carregar_ia():
    # Caminhos
    caminho_modelo = os.path.join("data", "models", "modelo_fake_news.pkl")
    caminho_vetor = os.path.join("data", "models", "vetorizador.pkl")
    modelo = joblib.load(caminho_modelo)
    vetorizador = joblib.load(caminho_vetor)
    return modelo, vetorizador

def prever():
    modelo, vetorizador = carregar_ia()
    print("\n================================================")
    print(" DETECTOR DE FAKE NEWS - COM ZONA DE INCERTEZA")
    print(" Digite 'sair' para encerrar.")
    print("================================================\n")
    
    while True:
        noticia = input(">>> Cole a notícia: ")
        if noticia.lower() == 'sair': break
            
        # Processamento
        limpo = limpar_texto(noticia)
        vetorizado = vetorizador.transform([limpo])
        
        # Pega as probabilidades [chance_real, chance_fake]
        probabilidades = modelo.predict_proba(vetorizado)[0]
        chance_fake = probabilidades[1] # Pega a chance de ser 1 (Fake)
        
        print("\n---------------- RESULTADO ----------------")
        
        # LÓGICA DA ZONA DE INCERTEZA (45% a 55%)
        if 0.45 <= chance_fake <= 0.55:
            print(f"⚠️ INCONCLUSIVO (Chance de Fake: {chance_fake*100:.2f}%)")
            print("   O modelo ficou em dúvida. O texto usa vocabulário neutro ou desconhecido.")
            
        elif chance_fake > 0.55:
            print(f"🚨 PARECE FAKE! (Confiança: {chance_fake*100:.2f}%)")
            
        else: # Menor que 0.45
            chance_real = 1 - chance_fake
            print(f"✅ PARECE REAL! (Confiança: {chance_real*100:.2f}%)")
            
        print("-------------------------------------------\n")

if __name__ == "__main__":
    prever()