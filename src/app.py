import streamlit as st
import joblib
import re
import string
import os
import numpy as np

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Detector de Fake News",
    page_icon="🕵️‍♂️",
    layout="centered"
)

# --- FUNÇÕES (Igual ao código original) ---
def limpar_texto(texto):
    if not isinstance(texto, str): return ""
    texto = texto.lower()
    texto = re.sub(r'https?://\S+|www\.\S+', '', texto)
    texto = re.sub(r'<.*?>', '', texto)
    texto = re.sub(r'[%s]' % re.escape(string.punctuation), '', texto)
    texto = re.sub(r'\n', ' ', texto)
    texto = re.sub(r'\w*\d\w*', '', texto)
    return texto

# Cache para não carregar o modelo toda vez que clicar no botão (deixa rápido)
@st.cache_resource
def carregar_ia():
    caminho_modelo = os.path.join("data", "models", "modelo_fake_news.pkl")
    caminho_vetor = os.path.join("data", "models", "vetorizador.pkl")
    modelo = joblib.load(caminho_modelo)
    vetorizador = joblib.load(caminho_vetor)
    return modelo, vetorizador

# --- INTERFACE VISUAL ---
st.title("🕵️‍♂️ Detector de Fake News")
st.markdown("---")

st.write("Cole a notícia abaixo para verificar a veracidade. Nossa IA analisará o padrão de escrita.")

# Área de texto
noticia = st.text_area("Manchete ou texto da notícia:", height=150, placeholder="Ex: O governo decretou o fim da internet...")

if st.button("🔍 Verificar Veracidade"):
    if not noticia.strip():
        st.warning("Por favor, digite algum texto antes de verificar.")
    else:
        # Carrega IA
        try:
            modelo, vetorizador = carregar_ia()
            
            # Processa
            limpo = limpar_texto(noticia)
            vetorizado = vetorizador.transform([limpo])
            probabilidades = modelo.predict_proba(vetorizado)[0]
            
            chance_fake = probabilidades[1]
            chance_real = probabilidades[0]
            
            # LÓGICA DE EXIBIÇÃO
            st.markdown("### Resultado da Análise:")
            
            # Zona de Incerteza (45% - 55%)
            if 0.45 <= chance_fake <= 0.55:
                st.warning("⚠️ **INCONCLUSIVO**")
                st.write(f"A IA ficou em dúvida (Chance de ser Fake: {chance_fake*100:.1f}%)")
                st.info("O texto utiliza vocabulário neutro ou desconhecido pelo modelo.")
                
            # É FAKE (> 55%)
            elif chance_fake > 0.55:
                st.error("🚨 **PARECE FAKE NEWS!**")
                st.metric(label="Probabilidade de ser Falso", value=f"{chance_fake*100:.1f}%")
                st.progress(int(chance_fake * 100)) # Barra de progresso vermelha (metaforicamente)
                
            # É REAL (< 45%)
            else:
                st.success("✅ **PARECE VERDADE!**")
                st.metric(label="Probabilidade de ser Real", value=f"{chance_real*100:.1f}%")
                st.progress(int(chance_real * 100)) # Barra de progresso verde
                
        except Exception as e:
            st.error(f"Erro ao carregar a IA. Verifique se os arquivos .pkl estão na pasta correta. Erro: {e}")

# Rodapé
st.markdown("---")
st.caption("Sistema desenvolvido para a disciplina de Inteligência Artificial. Modelo: Regressão Logística.")