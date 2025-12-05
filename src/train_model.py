import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def treinar_modelo():
    # Caminhos
    caminho_dados = os.path.join("data", "processed", "dados_limpos.csv")
    pasta_modelos = os.path.join("data", "models")
    
    # Cria a pasta de modelos, se não existir
    if not os.path.exists(pasta_modelos):
        os.makedirs(pasta_modelos)

    print("1. Carregando dados processados...")
    df = pd.read_csv(caminho_dados)
    
    # Garantir que não tem valores nulos (NaN) no texto
    df = df.dropna()

    X = df['texto_final'] # O texto (Features)
    y = df['label']       # A resposta (Target: 0 ou 1)

    # --- OS DOIS DATASETS ---
    # Dividimos em Treino (70%) e Teste (30%)
    # stratify=y garante que a proporção 50/50 de fakes/reais se mantenha nos dois grupos
    print("2. Dividindo em Treino e Teste (70/30)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # --- VETORIZAÇÃO (Texto -> Números) ---
    print("3. Vetorizando os textos (TF-IDF)...")
    # max_df=0.7 -> ignora palavras que aparecem em mais de 70% dos textos (são muito comuns)
    vectorizer = TfidfVectorizer(max_features=5000, max_df=0.7)
    
    # O vetorizador aprende o vocabulário apenas com o TREINO
    X_train_vec = vectorizer.fit_transform(X_train)
    # E depois apenas transforma o TESTE (sem aprender nada novo com ele)
    X_test_vec = vectorizer.transform(X_test)

    # --- TREINAMENTO ---
    print("4. Treinando o modelo (Regressão Logística)...")
    # fit_intercept=False diz para a IA: "Comece o com pesos em 0x0, e não em 3x0 pro Fake"
    model = LogisticRegression(fit_intercept=False)
    model.fit(X_train_vec, y_train)

    # --- AVALIAÇÃO ---
    print("5. Avaliando o modelo...")
    previsoes = model.predict(X_test_vec)
    acuracia = accuracy_score(y_test, previsoes)
    
    print(f"\n========================================")
    print(f"ACURÁCIA FINAL: {acuracia * 100:.2f}%")
    print(f"========================================\n")
    
    print("Relatório Detalhado:")
    print(classification_report(y_test, previsoes, target_names=['Real (0)', 'Fake (1)']))

    # --- SALVANDO TUDO ---
    print("6. Salvando o modelo e o vetorizador...")
    joblib.dump(model, os.path.join(pasta_modelos, "modelo_fake_news.pkl"))
    joblib.dump(vectorizer, os.path.join(pasta_modelos, "vetorizador.pkl"))
    print("Pronto! Modelo salvo na pasta 'data/models'.")

if __name__ == "__main__":
    treinar_modelo()