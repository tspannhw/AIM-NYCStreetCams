from __future__ import print_function
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
# import paho.mqtt.client as mqtt
import os
from pymilvus import connections
from pymilvus import utility
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import timm
from sklearn.preprocessing import normalize
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from pymilvus import MilvusClient
import os
from IPython.display import display

# -----------------------------------------------------------------------------

class FeatureExtractor:
    def __init__(self, modelname):
        # Load the pre-trained model
        self.model = timm.create_model(
            modelname, pretrained=True, num_classes=0, global_pool="avg"
        )
        self.model.eval()

        # Get the input size required by the model
        self.input_size = self.model.default_cfg["input_size"]

        config = resolve_data_config({}, model=modelname)
        # Get the preprocessing function provided by TIMM for the model
        self.preprocess = create_transform(**config)

    def __call__(self, imagepath):
        # Preprocess the input image
        input_image = Image.open(imagepath).convert("RGB")  # Convert to RGB if needed
        input_image = self.preprocess(input_image)

        # Convert the image to a PyTorch tensor and add a batch dimension
        input_tensor = input_image.unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            output = self.model(input_tensor)

        # Extract the feature vector
        feature_vector = output.squeeze().numpy()

        return normalize(feature_vector.reshape(1, -1), norm="l2").flatten()

extractor = FeatureExtractor("resnet34")

# Milvus Constants
COLLECTION_NAME = 'nycstreetcams'  # Collection name
DIMENSION = 512 # 2048  # Embedding vector size in this example
MILVUS_URL = "http://localhost:19530" 

# -----------------------------------------------------------------------------
# Connect to Milvus

# Milvus Lite
# milvus_client = MilvusClient(uri="streetcams.db")

# Local Docker Server
milvus_client = MilvusClient( uri=MILVUS_URL)

# 
query_image = "street1.png"

results = milvus_client.search(
    COLLECTION_NAME,
    data=[extractor(query_image)],
    output_fields=["filepath", "url", "videourl", "roadwayname", "name", "latlong", "directionoftravel", "id"],
    search_params={"metric_type": "COSINE"},
)
images = []
for result in results:
    print(result)
    for hit in result[:10]:
        filepath = hit["entity"]["filepath"]
        img = Image.open(filepath)
        print(filepath)
        print(hit["entity"]["roadwayname"])
 #       img = img.resize((150, 150))
 #       images.append(img)

#width = 150 * 5
#height = 150 * 2
#concatenated_image = Image.new("RGB", (width, height))

#for idx, img in enumerate(images):
#    x = idx % 5
#    y = idx // 5
#    concatenated_image.paste(img, (x * 150, y * 150))
#display("query")
#display(Image.open(query_image).resize((150, 150)))
#display("results")
#display(concatenated_image)
