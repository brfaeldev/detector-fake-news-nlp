# Detector de Fake News com Machine Learning 🕵️‍♂️

Projeto final desenvolvido para a disciplina de **Inteligência Artificial**. O objetivo é classificar notícias em "Reais" ou "Fakes" utilizando Processamento de Linguagem Natural (NLP) e Regressão Logística, com foco em interpretabilidade e eliminação de viés.

---

## 📊 Sobre o Dataset (Abordagem Híbrida)
Inicialmente, enfrentamos problemas de *overfitting* temporal (o modelo "decorou" o cenário político de 2018). Para corrigir isso e garantir generalização, utilizamos uma abordagem de **Data Augmentation** combinando dois datasets distintos:

1. **Fake.br-Corpus**
2. **FakeTrue.br**

Isso resultou em um **"Super Dataset" Balanceado** com **10.782 notícias**, rigorosamente dividido em:
* 🟢 **50% Notícias Reais**
* 🔴 **50% Notícias Falsas**

---

## 🛠️ Tecnologias e Metodologia Utilizadas no Projeto

### 🛠️ Tecnologias 
* **Python 3.13.7**
* **Pandas:** Manipulação de dados.
* **Scikit-Learn:** Machine Learning (Regressão Logística).
* **TF-IDF:** Vetorização de texto.
* **NLTK:** Processamento de texto e stopwords.
* **Streamlit:** 

### 🛠️ Metodologias 
* **Modelo:** Regressão Logística (*Logistic Regression*). Escolhido por ser um modelo "White Box" (transparente), permitindo auditoria de pesos e correção de vieses.
* **Vetorização:** TF-IDF (Term Frequency-Inverse Document Frequency).
* **Split de Treino/Teste:** 70% para Treino e 30% para Validação (com estratificação).
* **Lógica de Incerteza:** Implementação de uma "Zona de Dúvida". Se a confiança da IA ficar entre 45% e 55%, o sistema retorna **"Inconclusivo"** para evitar alucinações em temas desconhecidos.
* **Correção de Viés:** O modelo foi ajustado com `fit_intercept=False` para eliminar o preconceito estatístico inicial.

---

## 📂 Estrutura do Projeto

'''text
ia/
├── data/
│   ├── raw/            # Datasets originais (.csv)
│   ├── processed/      # Dataset unificado e limpo
│   ├── models/         # Modelo treinado (.pkl) e vetorizador
│   └── matriz_confusao.png # Gráfico de performance final
├── src/
│   ├── analise_inicial.py # Diagnóstico inicial dos dados brutos
│   ├── app.py             # Aplicação Web (Frontend com Streamlit)
│   ├── auditoria.py       # Validação de amostras aleatórias (Sanity Check)
│   ├── data_cleaning.py   # Script de limpeza e unificação dos datasets
│   ├── gerar_grafico.py   # Gera a visualização da Matriz de Confusão
│   ├── teste_manual.py    # Teste via terminal (sem interface gráfica)
│   ├── train_model.py     # Script de treinamento da IA (Treino/Teste split)
│   └── ver_pesos.py       # Diagnóstico de viés (pesos das palavras)
├── README.md              # Documentação do projeto
└── requirements.txt       # Lista de dependências
'''

---

## 🚀 Como Rodar o Projeto

1. **Instale as dependências:**
Certifique-se de ter o Python instalado e, no terminal, com a pasta do projeto selecionada, rode:

   ```bash
   pip install -r requirements.txt
   ```

Ou instale manualmente:

   ```bash
   pip install pandas numpy scikit-learn nltk matplotlib seaborn jupyter streamlit
   ```

2. **Preparação (Opcional - Já realizado)**
Os scripts de limpeza e treinamento (data_cleaning.py e train_model.py) já foram executados e os modelos estão salvos na pasta data/models. Não é necessário rodá-los novamente para testar.

3. **Executando a Aplicação Web**
Para abrir a interface visual e testar notícias em tempo real:

   ```bash
   streamlit run src/app.py
   ```
O navegador abrirá automaticamente com o sistema pronto para uso.

## 📈 Resultados Alcançados

O modelo final atingiu uma performance robusta no conjunto de teste (3.235 notícias nunca vistas pelo modelo):

* **Métrica**                    **Resultado**
* **Acurácia Global:**           91.53%
* **Precisão (Notícias Reais):** 97%
* **Precisão (Fake News):**      86%

Nota: O modelo prioriza a cautela. A taxa de falsos positivos (acusar uma verdade de ser mentira) foi reduzida a apenas ~2%, garantindo confiabilidade.