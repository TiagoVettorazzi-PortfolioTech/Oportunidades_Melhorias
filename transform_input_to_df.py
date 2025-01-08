import pandas as pd
import json
import logging
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load OpenAI API key from environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("API key for OpenAI is not set. Please configure it in your environment variables.")

# Initialize LLM with specific settings
def initialize_llm():
    return ChatOpenAI(
        temperature=0.1,
        model="gpt-4",  # Use um modelo válido
        openai_api_key=api_key,
    )

def transform_input_to_df(input_data):
    """
    Transform input data into a pandas DataFrame using the LLM directly.
    
    Parameters:
    input_data (str): Input data in the specified format.
    
    Returns:
    pandas.DataFrame: DataFrame with the transformed data.
    """
    try:
        # Initialize LLM
        llm = initialize_llm()

        # Define o prompt para garantir que a saída seja JSON puro
        prompt = f"""
        Você é um Desenvolvedor Sênior. Com base nos dados fornecidos, retorne-os em formato JSON válido.
        **Entrada:**
        {input_data}

        **Formato esperado (JSON):**
        [
            {{
                "Oportunidade de Melhoria": "Descrição clara da oportunidade",
                "Solução": "Solução recomendada",
                "Backlog de Atividades": "Atividades sugeridas",
                "Investimento": "Investimento necessário",
                "Ganhos": "Ganhos esperados"
            }}
        ]

        Não inclua explicações adicionais, apenas o JSON válido.
        """

        # Chama o modelo com o prompt
        response = llm.invoke(prompt)

        # print(response)

        # Valida e carrega o JSON
        try:
            data = json.loads(response.content)
        except json.JSONDecodeError:
            logger.error("Falha ao analisar JSON. Verifique a saída.")
            raise ValueError(f"Resposta inválida: {response}")

        # Converte o JSON para DataFrame
        df = pd.DataFrame(data)
        return df
    
    except Exception as e:
        logger.error("Error during transformation: %s", e)
        return f"Error during transformation: {str(e)}"

# # Testando o script
# if __name__ == "__main__":
#     test_input = """
#     [
#         {
#             "Oportunidade de Melhoria": "Exemplo 1",
#             "Solução": "Implementação de XYZ",
#             "Backlog de Atividades": "- Planejamento\\n- Execução",
#             "Investimento": "1000 USD",
#             "Ganhos": "Aumento de 20% em eficiência"
#         }
#     ]
#     """

#     df = transform_input_to_df(test_input)
#     print(df)
