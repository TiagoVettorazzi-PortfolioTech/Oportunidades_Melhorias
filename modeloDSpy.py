from weaviate import Client
from weaviate.auth import AuthApiKey
from weaviate.classes.init import Auth
import os
from dotenv import load_dotenv
import dspy
from dspy.retrieve.weaviate_rm import WeaviateRM
import weaviate
from openai import OpenAI
from dspy_DocsOportune import OportuneRAG
import streamlit as st
from weaviate.client import WeaviateClient

class OportuneRAGClient:
    def __init__(self):
        # Explicitly load .env file and print debug information
        load_dotenv()

        # Retrieve environment variables
        self.secretk = os.environ.get("OPENAI_API_KEY")
        self.weaviate_cluster_url = os.environ.get("WEAVIATE_URL")
        self.weaviate_api_key = os.environ.get("WEAVIATE_API_KEY")

        self.client = OpenAI(api_key=self.secretk)
        self.weaviate_client = self.setup_weaviate_client()
        self.params4o = self.setup_dspy_params()
        self.modelo = OportuneRAG()
 
    def setup_weaviate_client(self):
        try:
            # Validar URL do cluster
            if not self.weaviate_cluster_url:
                raise ValueError("Weaviate cluster URL is not set")

            # Criar cliente Weaviate
            weaviate_client = weaviate.connect_to_weaviate_cloud(
                cluster_url=self.weaviate_cluster_url,
                auth_credentials = Auth.api_key(self.weaviate_api_key),
                headers = {"X-OpenAI-Api-Key": self.secretk}
            )


            # Testar conexão
            # result = weaviate_client.data_object.get(class_name='DocsOportunidades', limit=3)
            # print(f"Consulta bem-sucedida: {result}")

            return weaviate_client
        
        except Exception as e:
            print(f"ERRO CONEXAO: {e}")
            print(f"Cluster URL: {self.weaviate_cluster_url}")
            print(f"API Key provided: {bool(self.weaviate_api_key)}")
            return None
        
    def setup_dspy_params(self):
        try:
            if self.weaviate_client is None:
                raise ValueError("Weaviate client is not initialized")
            # print(dir(self.weaviate_client))
            return {
                "lm": dspy.OpenAI(model='gpt-4o', max_tokens=2048, temperature=0.2),
                "rm": WeaviateRM("DocsOportunidades",
                                 weaviate_client=self.weaviate_client, 
                                 # PASSAMOS O COLLECTION TEXT KEY QUE SERA USADO PARA CLASSIFICAR O TEXTO => (CONTEXT)
                                 weaviate_collection_text_key = "sOLUCAO"
                                 )
            }
        except Exception as e:
            print(f"ERRO DSpy.settings: {e}")
            return None

    def close_weaviate_client(self):
        try:
            if self.weaviate_client.is_connected():
                self.weaviate_client.close()
        except Exception as e:
            print(f"ERRO AO FECHAR WEAVIATE CLIENT: {e}")

    def run_model(self, prompt):
        try:
            
            # Check if params are properly set
            if self.params4o is None:
                raise ValueError("DSpy parameters not properly initialized")
            dspy.settings.configure(**self.params4o)
            resposta = self.modelo(question=prompt)
            self.close_weaviate_client()
            return resposta.answer
        
        except Exception as e:
            print(f"ERRO AO RODAR O MODELO: {e}")
            self.close_weaviate_client()
            return None


