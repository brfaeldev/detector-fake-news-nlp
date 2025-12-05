import pandas as pd
import re
import string
import nltk
import os
import glob

nltk.download('stopwords')
nltk.download('punkt')

def limpar_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r'https?://\S+|www\.\S+', '', texto)
    texto = re.sub(r'<.*?>', '', texto)
    texto = re.sub(r'[%s]' % re.escape(string.punctuation), '', texto)
    texto = re.sub(r'\n', ' ', texto)
    texto = re.sub(r'\w*\d\w*', '', texto)
    return texto

def padronizar_dataset(df, nome_arquivo):
    print(f"   -> Processando: {nome_arquivo} | Colunas: {list(df.columns)}")
    
    colunas_lower = [c.lower() for c in df.columns]
    
    # CASO ESPECÍFICO: Dataset que tem colunas 'fake' e 'true' separadas
    # (É o caso do faketruebrcorpus.csv)
    if 'fake' in colunas_lower and 'true' in colunas_lower and 'label' not in colunas_lower:
        print("      [Detecção] Formato de colunas separadas (fake/true) encontrado.")
        
        # Pega as fakes
        df_fake = df[['fake']].copy()
        df_fake = df_fake.rename(columns={'fake': 'texto_original'})
        df_fake['label'] = 1 # Fake é 1
        
        # Pega as reais
        df_real = df[['true']].copy()
        df_real = df_real.rename(columns={'true': 'texto_original'})
        df_real['label'] = 0 # Real é 0
        
        # Junta tudo
        df_unificado = pd.concat([df_fake, df_real], axis=0)
        return df_unificado.dropna()

    # CASO PADRÃO: Tem uma coluna de texto e uma de label
    col_label = None
    possiveis_labels = ['label', 'classe', 'class', 'target']
    for col in df.columns:
        if col.lower() in possiveis_labels:
            col_label = col
            break
            
    col_texto = None
    possiveis_textos = ['preprocessed_news', 'text', 'texto', 'news', 'conteudo', 'body']
    for col in df.columns:
        if col.lower() in possiveis_textos:
            col_texto = col
            break
            
    if col_label and col_texto:
        df = df.rename(columns={col_label: 'label', col_texto: 'texto_original'})
        df = df[['texto_original', 'label']]
        
        # Normaliza labels de texto para número
        if df['label'].dtype == 'object':
            mapa = {'fake': 1, 'true': 0, 'falso': 1, 'verdadeiro': 0, 'real': 0}
            df['label'] = df['label'].str.lower().map(mapa)
            
        return df.dropna()

    print(f"   ⚠️ AVISO: Não consegui entender as colunas de {nome_arquivo}. Pulando.")
    return None

def preparar_dados():
    caminho_raw = os.path.join("data", "raw")
    caminho_saida = os.path.join("data", "processed", "dados_limpos.csv")
    
    arquivos_csv = glob.glob(os.path.join(caminho_raw, "*.csv"))
    
    if not arquivos_csv:
        print("ERRO: Nenhum arquivo .csv encontrado em data/raw")
        return

    dfs_para_juntar = []
    
    for arquivo in arquivos_csv:
        try:
            # encoding='utf-8' as vezes dá erro no windows, se der tente 'latin-1'
            df_temp = pd.read_csv(arquivo, encoding='utf-8', on_bad_lines='skip')
            
            df_padronizado = padronizar_dataset(df_temp, os.path.basename(arquivo))
            
            if df_padronizado is not None:
                dfs_para_juntar.append(df_padronizado)
                print(f"      -> Sucesso! +{len(df_padronizado)} linhas.")
        except Exception as e:
            print(f"   -> Erro ao ler {arquivo}: {e}")

    if dfs_para_juntar:
        df_final = pd.concat(dfs_para_juntar, axis=0, ignore_index=True)
        print("------------------------------------------------")
        print(f"TOTAL FINAL COMBINADO: {len(df_final)} notícias.")
        
        # Diagnóstico de Balanceamento
        contagem = df_final['label'].value_counts()
        print(f"Distribuição: {contagem.to_dict()} (0=Real, 1=Fake)")
        
        print("Aplicando limpeza no texto (pode demorar)...")
        df_final['texto_final'] = df_final['texto_original'].apply(limpar_texto)
        df_final = df_final[df_final['texto_final'].str.strip() != '']
        
        df_final[['texto_final', 'label']].to_csv(caminho_saida, index=False)
        print(f"Arquivo salvo em: {caminho_saida}")
    else:
        print("Nenhum dado foi processado.")

if __name__ == "__main__":
    preparar_dados()