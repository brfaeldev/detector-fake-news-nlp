import pandas as pd
import os

def auditoria_final():
    caminho = os.path.join("data", "processed", "dados_limpos.csv")
    df = pd.read_csv(caminho)
    
    print(f"Total de dados: {len(df)}")
    
    # Pegamos 3 exemplos aleatórios que estão marcados como FAKE (1)
    print("\n=== AUDITORIA: O QUE A IA APRENDEU QUE É FAKE (1)? ===")
    print("(Se você ler verdades aqui, estamos ferrados)")
    amostra_fake = df[df['label'] == 1].sample(3)
    for i, row in amostra_fake.iterrows():
        print(f"\n[LABEL {row['label']}] Texto: {row['texto_final'][:150]}...")
        
    # Pegamos 3 exemplos aleatórios que estão marcados como REAL (0)
    print("\n=======================================================")
    print("=== AUDITORIA: O QUE A IA APRENDEU QUE É REAL (0)? ===")
    print("(Se você ler loucuras aqui, estamos ferrados)")
    amostra_real = df[df['label'] == 0].sample(3)
    for i, row in amostra_real.iterrows():
        print(f"\n[LABEL {row['label']}] Texto: {row['texto_final'][:150]}...")

if __name__ == "__main__":
    auditoria_final()