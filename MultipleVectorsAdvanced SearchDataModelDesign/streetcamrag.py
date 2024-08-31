import os
from pymilvus import MilvusClient
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama
from langchain import hub
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus
from langchain import hub
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import requests
import sys
import io
import json
import shutil
import sys
import datetime
import subprocess
import sys
import os
import math
import base64
from time import gmtime, strftime
import random, string
import time
import psutil
import base64
import uuid
import socket
import os
from pymilvus import connections
from pymilvus import utility
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection
import torch
from torchvision import transforms
from PIL import Image
import timm
from sklearn.preprocessing import normalize
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from IPython.display import display
from pymilvus import MilvusClient
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv
load_dotenv(verbose=True)
import json
import requests
from datetime import datetime, timedelta
from langchain_milvus import Milvus
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv
load_dotenv(verbose=True)
from langchain_core.globals import set_verbose, set_debug

# Disable verbose logging
set_verbose(False)

# Disable debug logging
set_debug(False)

### Setup environment and constants

DIMENSION = 384 
MILVUS_URL = "http://192.168.1.153:19530" 

slack_token = os.environ.get("SLACK_BOT_TOKEN")
client = WebClient(token=slack_token)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
SC_COLLECTION_NAME = "nycstreetcameras"
TODAYS_DATE = str( datetime.today().strftime('%Y-%m-%d') ) 
YESTERDAYS_DATE = (datetime.now() - timedelta(1)).strftime('%Y-%m-%d')

### Environment Variables needed
os.environ["LANGCHAIN_HUB_API_URL"] = "https://api.hub.langchain.com"
os.environ["LANGCHAIN_HUB_API_KEY"] = os.environ.get("LANGCHAIN_HUB_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_HUB_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.hub.langchain.com"

### Turn off slack warnings
os.environ["SKIP_SLACK_SDK_WARNING"] = "false"

# https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.milvus.Milvus.html 

### Set up connection to Milvus for NYC Street Cameras, weather text vector
vector_store = Milvus(
    embedding_function=embeddings,
    collection_name=SC_COLLECTION_NAME,
    primary_field = "id",
    vector_field = "weather_text_vector",
    text_field="weatherdetails",
    connection_args={"uri": MILVUS_URL},
)

### Run Ollama Against LLAMA3
def run_query() -> None:
    llm = Ollama(
        model="llama3",
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        stop=["<|eot_id|>"],
    )

    query = input("\nQuery: ")
    qa_chain = RetrievalQA.from_chain_type(
        llm, retriever=vector_store.as_retriever(collection =  SC_COLLECTION_NAME))

    result = qa_chain({"query": query})
    resultforslack = str(result["result"])
    print(resultforslack)

    try:
        response = client.chat_postMessage(mrkdwn=True, channel="C06NE1FU6SE", text="", 
                                            blocks=[{"type": "section","text": {"type": "mrkdwn","text": "*" + str(query) + "*  \n\n" + str(resultforslack) +"\n" }}])

    
    except SlackApiError as e:
        # You will get a SlackApiError if "ok" is False
        print("Slack failed")

if __name__ == "__main__":
    while True:
        run_query()
