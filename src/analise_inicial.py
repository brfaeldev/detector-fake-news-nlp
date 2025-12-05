import pandas as pd
import os

# Caminho do arquivo
caminho_arquivo = os.path.join("data", "raw", "fakebrcorpus.csv")

try:
    # Tenta ler o arquivo
    df = pd.read_csv(caminho_arquivo)
    
    print("=== Relatório do Arquivo ===")
    print(f"Total de linhas: {len(df)}")
    print(f"Colunas encontradas: {df.columns.tolist()}")
    
    # Descobrindo qual coluna é a de 'Fake ou Real'
    # Geralmente é 'label', 'classe', 'fake' etc.
    if 'label' in df.columns:
        contagem = df['label'].value_counts()
        print("\nDivisão das Classes (0 e 1):")
        print(contagem)
        
        if len(contagem) < 2:
            print("\nALERTA: Parece que esse arquivo só tem UM tipo de notícia (só fake ou só real).")
            print("Precisaremos de outro dataset para complementar.")
        else:
            print("\nÓtimo! O arquivo já contém Reais e Fakes misturados.")
            
    else:
        print("\nNão achei uma coluna chamada 'label'. Veja os nomes das colunas acima e me diga qual parece ser a classificação.")

    # Mostra as primeiras linhas para a gente ver os dados
    print("\n=== Exemplo de Dados ===")
    print(df.head(2))

except FileNotFoundError:
    print(f"Erro: Não achei o arquivo em {caminho_arquivo}. Verifique se o nome está exato.")
except Exception as e:
    print(f"Ocorreu um erro: {e}")