import pandas as pd
import openai
import weaviate
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Configuração da API OpenAI
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Conexão com o Weaviate
client_wv8 = weaviate.Client(
    url=os.environ.get("WEAVIATE_URL"),
    auth_client_secret=weaviate.AuthApiKey(api_key=os.environ.get("WEAVIATE_API_KEY")),
    additional_headers={
        "X-OpenAI-Api-Key": openai.api_key,
    }
)

# Criar o schema dinamicamente
schema = {
    "classes": [
        {
            "class": "DocsOportunidades",
            "description": "Detalhes de melhorias em processos empresariais",
            "properties": [
                {"name": "EMPRESA", "dataType": ["text"], "description": "Nome da empresa"},
                {"name": "SEGMENTO_MERCADO", "dataType": ["text"], "description": "Segmento de mercado"},
                {"name": "PROCESSO", "dataType": ["text"], "description": "Descrição do processo"},
                {"name": "ATIVIDADE_RELACIONADA", "dataType": ["text"], "description": "Atividade relacionada ao processo"},
                {"name": "MELHORIA_SUGERIDA", "dataType": ["text"], "description": "Categoria da melhoria sugerida"},
                {"name": "GAPS", "dataType": ["text"], "description": "Problemas ou lacunas identificadas"},
                {"name": "CAUSA", "dataType": ["text"], "description": "Causa raiz do problema"},
                {"name": "SOLUCAO", "dataType": ["text"], "description": "Proposta de melhoria ou solução"},
                {"name": "GANHOS", "dataType": ["text"], "description": "Ganhos ou objetivos esperados com a solução"}
            ],
            "vectorizer": "text2vec-openai"
        }
    ]
}

# Verificar se o schema existe, se não, criá-lo
class_name = "DocsOportunidades"
existing_classes = [cls["class"] for cls in client_wv8.schema.get().get("classes", [])]

if class_name not in existing_classes:
    client_wv8.schema.create(schema)
    print(f"Schema '{class_name}' criado com sucesso.")
else:
    print(f"Schema '{class_name}' já existe.")

# Carregar os dados
df_base = pd.read_excel('Base.xlsx', skiprows=2, header=1)
df_base = df_base.iloc[:, :9]
df_base = df_base.fillna("Não informado")

renomear = {
    "EMPRESA": "EMPRESA",
    "SEGMENTO DE MERCADO": "SEGMENTO_MERCADO",
    "PROCESSO": "PROCESSO",
    "ATIVIDADE RELACIONADA": "ATIVIDADE_RELACIONADA",
    "TIPO DE MELHORIA": "MELHORIA_SUGERIDA",
    "DESCONEXÕES (GAP)": "GAPS",
    "CAUSA": "CAUSA",
    "MELHORIA/SOLUÇÃO": "SOLUCAO",
    "GANHOS/OBJETIVO": "GANHOS"
}

df_base = df_base.rename(columns=renomear)

# Preparar os dados para inserção
lista = df_base.to_dict(orient='records')

# Inserir os dados no Weaviate
try:
    client_wv8.batch.configure(batch_size=100)
    with client_wv8.batch as batch:
        for i, d in enumerate(lista):
            properties = {
                "EMPRESA": d['EMPRESA'],
                "SEGMENTO_MERCADO": d['SEGMENTO_MERCADO'],
                "PROCESSO": d['PROCESSO'],
                "ATIVIDADE_RELACIONADA": d['ATIVIDADE_RELACIONADA'],
                "MELHORIA_SUGERIDA": d['MELHORIA_SUGERIDA'],
                "GAPS": d['GAPS'],
                "CAUSA": d['CAUSA'],
                "SOLUCAO": d['SOLUCAO'],
                "GANHOS": d['GANHOS']
            }
            try:
                batch.add_data_object(
                    data_object=properties,
                    class_name="DocsOportunidades"
                )
                print(f"Registro {i + 1} importado com sucesso.")
            except Exception as e:
                print(f"Erro ao adicionar o registro {i + 1}: {e}")
except Exception as e:
    print(f"Erro ao iniciar o batch: {e}")
