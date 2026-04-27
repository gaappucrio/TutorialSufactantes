Predição de Fases de Surfactantes com Machine Learning 🧪💻
Este repositório contém uma infraestrutura completa de Engenharia de Dados (ETL) e Machine Learning desenvolvida para prever o comportamento de fase de surfactantes (Isotrópica, Lamelar, Hexagonal, etc.) com base nas leis da termodinâmica (Temperatura e Concentração).

O pipeline extrai dados brutos de um banco de dados JSON complexo (derivado de pesquisas publicadas na Nature), converte as matrizes químicas em uma tabela estruturada e treina um classificador inteligente.

🛠️ 1. Pré-requisitos e Configuração do Ambiente
Para garantir que todos do grupo rodem o código sem o famoso erro "na minha máquina funciona", padronizamos o ambiente utilizando o WSL2 (Ubuntu) e o pyenv com a versão exata do Python.

Instalação das dependências do sistema:
Abra o terminal do WSL e execute:

```bash
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git
```
Configuração do Python (via pyenv):

```bash
# Instala a versão correta para evitar erros de sintaxe (como nas f-strings)
pyenv install 3.12.3

# Cria o ambiente virtual isolado para o projeto
pyenv virtualenv 3.12.3 projeto_surfactantes

# Ativa o ambiente
pyenv activate projeto_surfactantes

# Instala as bibliotecas de processamento e Inteligência Artificial
pip install pandas numpy scikit-learn xgboost matplotlib
🔄 2. O Pipeline de Dados (ETL)
O banco de dados original armazena os experimentos químicos em um formato JSON aninhado, onde as medições são salvas como strings de texto. O script de ETL (extrator_etl.py) abre esse banco, interpreta a matemática e "achata" os dados em um CSV estruturado.

Crie o arquivo extrator_etl.py e rode com python extrator_etl.py:

Python
import json
import ast
import pandas as pd

# Arquivo fonte (pode ser o demo ou o banco real completo)
caminho_json = 'demo_data/sample_db.json'
dados_processados = []

print("Iniciando o processo de ETL...")

with open(caminho_json, 'r', encoding='utf-8') as f:
    dados_brutos = json.load(f)

for molecula_id, info in dados_brutos.items():
    if 'Data' not in info:
        continue

    smiles = info.get('SMILES', 'Desconhecido')
    dados_exp = info['Data']

    try:
        temperaturas = ast.literal_eval(dados_exp.get('temperature_(degC)', '[]'))
        composicoes = ast.literal_eval(dados_exp.get('composition_(wt%)', '[]'))
    except (ValueError, SyntaxError):
        continue 

    fases_chaves = [k for k in dados_exp.keys() if k not in ['temperature_(degC)', 'composition_(wt%)']]
    
    probabilidades_fases = {}
    for fase in fases_chaves:
        probabilidades_fases[fase] = ast.literal_eval(dados_exp[fase])

    qtd_pontos = len(temperaturas)
    for i in range(qtd_pontos):
        temp_atual = temperaturas[i]
        comp_atual = composicoes[i]
        
        fase_dominante = None
        maior_prob = -1.0
        
        for fase in fases_chaves:
            prob_atual = probabilidades_fases[fase][i]
            if prob_atual > maior_prob:
                maior_prob = prob_atual
                fase_dominante = fase
        
        if maior_prob > 0:
            dados_processados.append({
                'SMILES': smiles,
                'Temperatura': temp_atual,
                'Concentracao': comp_atual,
                'Fase_Dominante': fase_dominante
            })

df = pd.DataFrame(dados_processados)
df.to_csv('dados_limpos_surfactantes.csv', index=False)
print(f"Sucesso! Gerados {len(df)} pontos no arquivo 'dados_limpos_surfactantes.csv'.")
🧠 3. Treinamento da Inteligência Artificial
Com o CSV em mãos, a IA entra em ação. O script abaixo (treinar_modelo.py) foi blindado para rodar perfeitamente tanto com poucos dados (testes rápidos) quanto com milhares de registros (treinamento real).

Execute com python treinar_modelo.py:

Python
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

caminho_csv = 'dados_limpos_surfactantes.csv'
df = pd.read_csv(caminho_csv)
print(f"Total de amostras: {len(df)}")

X = df[['Temperatura', 'Concentracao']]

encoder = LabelEncoder()
y = encoder.fit_transform(df['Fase_Dominante'])

# Prevenção de erro matemático em datasets muito pequenos (modo demo)
if len(df) < 50:
    print("Modo Demo: Pulando split para não quebrar as classes.")
    X_train, X_test, y_train, y_test = X, X, y, y
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Treinando o modelo...")
modelo = xgb.XGBClassifier(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=5, 
    objective='multi:softprob',
    num_class=len(encoder.classes_),
    random_state=42
)

modelo.fit(X_train, y_train)

previsoes = modelo.predict(X_test)
acuracia = accuracy_score(y_test, previsoes)

print("\n=== Resultado Científico ===")
print(f"Acurácia: {acuracia * 100:.2f}%\n")
print(classification_report(y_test, previsoes, target_names=encoder.classes_))
⚙️ Como Customizar o Projeto (Guia para o Time)
A infraestrutura foi pensada para ser modular. O seu grupo pode alterar livremente os dados e as técnicas aplicadas.

Como usar o Banco de Dados Real (Gigante)
Baixe o arquivo de dados original fornecido pelos autores do estudo (ex: PhDat.json).

Coloque o arquivo dentro da pasta do projeto.

No arquivo extrator_etl.py, altere a linha 6:

Mude de: caminho_json = 'demo_data/sample_db.json'

Para: caminho_json = 'nome_do_seu_arquivo_gigante.json'

Rode o python extrator_etl.py novamente. O script fará o parsing de todo o banco automaticamente.

Como testar outros Modelos de IA (Além do XGBoost)
Se quiserem comparar a performance com Redes Neurais (MLP) ou Árvores de Decisão (Random Forest), basta mudar a inicialização no arquivo treinar_modelo.py.

Exemplo: Trocando para o Random Forest
No topo do arquivo, adicione o import:

Python
from sklearn.ensemble import RandomForestClassifier
Vá até a parte onde o XGBoost é inicializado e substitua por:

Python
modelo = RandomForestClassifier(
    n_estimators=100, 
    max_depth=5, 
    random_state=42
)
O resto do pipeline (treinamento e avaliação) continua funcionando da mesma forma! A arquitetura Scikit-Learn facilita muito esse tipo de teste "plug-and-play".
