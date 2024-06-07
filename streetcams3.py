from __future__ import print_function
import requests
from ultralytics import YOLO
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
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from pymilvus import connections
from pymilvus import utility
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection
import glob
import torch
from torchvision import transforms
from PIL import Image
import timm
from sklearn.preprocessing import normalize
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from pymilvus import MilvusClient
import os

# -----------------------------------------------------------------------------
#shutil.rmtree('runs/detect')

# NYC URL for Street Cams List
url = os.environ["NYURL"]

# Milvus Constants
COLLECTION_NAME = 'nycstreetcams'  # Collection name
DIMENSION = 512 # 2048  # Embedding vector size in this example
MILVUS_URL = "http://localhost:19530" 

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

# -----------------------------------------------------------------------------
# Access NYC 511 to Get List of Cameras
response = requests.get(url).content

# -----------------------------------------------------------------------------
# json format for NYC result
json_object = json.loads(response)

# Intialize
latitude = ""
longitude = ""
latlong = ""
strid = ""
strname = ""
directionoftravel = ""
url = ""
videourl = ""
roadwayname = ""

# -----------------------------------------------------------------------------
# ultralytics Yolo v8 Model

yolomodel = YOLO('yolov8n.pt')  # pretrained YOLOv8n model

# Slack client
slack_token = os.environ["SLACK_BOT_TOKEN"]
client = WebClient(token=slack_token)

# -----------------------------------------------------------------------------
# Connect to Milvus

# Milvus Lite
# milvus_client = MilvusClient(uri="streetcams.db")

# Local Docker Server
milvus_client = MilvusClient( uri=MILVUS_URL)

# -----------------------------------------------------------------------------
# Create collection which includes the id, filepath of the image, and image embedding
fields = [
    FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name='latlong', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='name', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='roadwayname', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='directionoftravel', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='videourl', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='url', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='filepath', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='vector', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
]

schema = CollectionSchema(fields=fields)

milvus_client.create_collection(COLLECTION_NAME, DIMENSION, schema=schema, metric_type="COSINE", auto_id=True)

index_params = milvus_client.prepare_index_params()
index_params.add_index(field_name = "vector", metric_type="COSINE")

milvus_client.create_index(COLLECTION_NAME, index_params)

# -----------------------------------------------------------------------------
# Iterate json urls
for jsonitems in json_object:
    if (  not jsonitems['Disabled'] and not jsonitems['Blocked'] ):
        print(jsonitems['Name'])
        latitude = jsonitems['Latitude']
        longitude = jsonitems['Longitude']
        latlong = str(latitude) + "," + str(longitude)
        strid = jsonitems['ID']
        strname = jsonitems['Name']
        directionoftravel = jsonitems['DirectionOfTravel']
        roadwayname = jsonitems['RoadwayName']
        url = jsonitems['Url']
        videourl = jsonitems['VideoUrl']
        uuid2 = "{0}_{1}".format(strftime("%Y%m%d%H%M%S",gmtime()),uuid.uuid4())
        url = str(url) + "#" + str(uuid2)
        img = requests.get(url)
        strfilename = str(uuid2) + ".png"
        filepath = "/opt/demo/nifi-2.0.0-M2/camimages/" + strfilename
        if img.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(img.content)

        results = yolomodel.predict(filepath, stream=False, save=True, imgsz=640, conf=0.5)

# -----------------------------------------------------------------------------
# Iterate results
        for result in results:
            outputimage = result.path
            savedir = result.save_dir
            speed = result.speed
            names = result.names
            boxes = result.boxes  # Boxes object for bounding box outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs
            obb = result.obb  # Oriented boxes object for OBB outputs
            # result.show()  # display to screen
            resultfilename = "/opt/demo/nifi-2.0.0-M2/camimages/{0}.png".format(uuid.uuid4())
            result.save(filename=resultfilename)  # save to disk
            strText = ":tada:" + str(strname) + ":" + str(roadwayname)

# -----------------------------------------------------------------------------
# Slack
            # TODO:  look for text on No live camera
            try:
                response = client.chat_postMessage(
                    channel="C06NE1FU6SE",
                    text=strText
                )
            except SlackApiError as e:
                # You will get a SlackApiError if "ok" is False
                assert e.response["error"]    # str like 'invalid_auth', 'channel_not_found'

            try:
                response = client.files_upload_v2(
                    channel="C06NE1FU6SE",
                    file=filepath,
                    title=roadwayname,
                    initial_comment="Traffic camera original image " + str(strfilename),
                )
            except SlackApiError as e:
                assert e.response["error"]

            try:
                response = client.files_upload_v2(
                    channel="C06NE1FU6SE",
                    file=resultfilename,
                    title=roadwayname,
                    initial_comment="Transformed image " + str(strfilename),
                )
            except SlackApiError as e:
                assert e.response["error"]


# -----------------------------------------------------------------------------
# Milvus insert
            try:
                imageembedding = extractor(resultfilename)
                milvus_client.insert( COLLECTION_NAME, {"vector": imageembedding, "filepath": filepath, "url": url, "videourl": videourl, "latlong": latlong, "name": strname, "roadwayname": roadwayname,"directionoftravel": directionoftravel, "videourl": videourl})

                print("resultfilename:" + resultfilename)
                print("Milvus:sent collection:" + roadwayname)

            except Exception as e:
                print("An error:", e)

# -----------------------------------------------------------------------------
