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
from tqdm import tqdm

# https://milvus.io/docs/integrate_with_pytorch.md
# -----------------------------------------------------------------------------
#shutil.rmtree('runs/detect')

COLLECTION_NAME = 'image_search'  # Collection name
DIMENSION = 2048  # Embedding vector size in this example
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

# Inference Argument
BATCH_SIZE = 128
TOP_K = 3

url = os.environ["NYURL"]

# -----------------------------------------------------------------------------
# Torch - Preprocessing
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# -----------------------------------------------------------------------------
# Connect to Milvus
connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)

# -----------------------------------------------------------------------------
# Access NYC 511 to Get List of Cameras
response = requests.get(url).content

# -----------------------------------------------------------------------------
# json format for NYC result
json_object = json.loads(response)

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
# Slack client
model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model
slack_token = os.environ["SLACK_BOT_TOKEN"]
client = WebClient(token=slack_token)

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
    FieldSchema(name='image_embedding', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
]
schema = CollectionSchema(fields=fields)
collection = Collection(name=COLLECTION_NAME, schema=schema)

# Create an AutoIndex index for collection
index_params = {
'metric_type':'L2',
'index_type':"IVF_FLAT",
'params':{'nlist': 16384}
}
collection.create_index(field_name="image_embedding", index_params=index_params)
collection.load()

# -----------------------------------------------------------------------------
# Iterate json urls
for jsonitems in json_object:
    print(jsonitems['Name'])
    # print("Disabled:" + str(jsonitems['Disabled']))
    # print("Blocked:" + str(jsonitems['Blocked']))

    if (  not jsonitems['Disabled'] and not jsonitems['Blocked'] ):
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

        # print ("URL:" + str(url))

        url = str(url) + "#" + str(uuid2)
        img = requests.get(url)
        strfilename = str(uuid2) + ".png"
        filepath = "/opt/demo/nifi-2.0.0-M2/camimages/" + strfilename
        if img.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(img.content)

        results = model.predict(filepath, stream=False, save=True, imgsz=640, conf=0.5)

# -----------------------------------------------------------------------------
# Iterate results
        for result in results:
            # print(result)
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
            # jsonoutput = result.tojson()
            # print("names:" + json.dumps(names))
            # print("speed:" + json.dumps(speed))
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
# Milvus Embed
            ### Emed
            try:
                print("resultfilename:" + resultfilename)
                im = Image.open(resultfilename).convert('RGB')

                data_batch = [[],[]]
                data_batch[0].append(preprocess(im))
                data_batch[1].append(outputimage)

                with torch.no_grad():
                    output = model(torch.stack(data_batch[0])).squeeze()

                    # print(output)

                    collection.insert([latlong, strname, roadwayname, directionoftravel, videourl, url, outputimage, output.tolist()])
                    data_batch = [[],[]]
                    collection.flush()
                    print("sent collection")

            except Exception as e:
                print("An error:", e)

# -----------------------------------------------------------------------------
