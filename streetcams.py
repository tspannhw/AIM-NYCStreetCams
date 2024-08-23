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
from pymilvus import model
from pymilvus.model.dense import SentenceTransformerEmbeddingFunction

model = SentenceTransformerEmbeddingFunction('all-MiniLM-L6-v2',device='cpu' )

# -----------------------------------------------------------------------------
#shutil.rmtree('runs/detect')

# NYC URL for Street Cams List
url = os.environ["NYURL"]

# Milvus Constants
COLLECTION_NAME = 'nycstreetcameras'  # Collection name
MILVUS_URL = "http://192.168.1.153:19530" 

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

#------ weather lookup
def weatherparse(url):
    weatherjson = requests.get(url).content
    weatherjsonobject = json.loads(weatherjson)

    weatherfields = {}
    weatherfields['creationdate'] = str(weatherjsonobject['creationDate'])

    locationfields = weatherjsonobject['location']

    weatherfields['areadescription'] = str(locationfields['areaDescription'])
    weatherfields['elevation'] = str(locationfields['elevation'])
    weatherfields['county'] = str(locationfields['county'])
    weatherfields['metar'] = str(locationfields['metar'])

    currentobservation = weatherjsonobject['currentobservation']

    weatherfields['weatherid']= str(currentobservation['id'])
    weatherfields['weathername'] = str(currentobservation['name'])
    weatherfields['observationdate'] = str(currentobservation['Date'])
    weatherfields['temperature'] = str(currentobservation['Temp'])
    weatherfields['dewpoint'] = str(currentobservation['Dewp'])
    weatherfields['relativehumidity'] = str(currentobservation['Relh'])
    weatherfields['windspeed'] = str(currentobservation['Winds'])
    weatherfields['winddirection'] = str(currentobservation['Windd'])
    weatherfields['gust'] = str(currentobservation['Gust'])
    weatherfields['weather'] = str(currentobservation['Weather'])
    weatherfields['visibility'] = str(currentobservation['Visibility'])
    weatherfields['altimeter'] = str(currentobservation['Altimeter'])
    weatherfields['slp'] = str(currentobservation['SLP'])
    weatherfields['timezone'] = str(currentobservation['timezone'])
    weatherfields['state'] = str(currentobservation['state'])
    weatherfields['windchill'] = str(currentobservation['WindChill'])
    
    return weatherfields
    

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
milvus_client = MilvusClient( uri=MILVUS_URL )

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
    FieldSchema(name='creationdate', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='areadescription', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='elevation', dtype=DataType.VARCHAR, max_length=200), 
    FieldSchema(name='county', dtype=DataType.VARCHAR, max_length=200), 
    FieldSchema(name='metar', dtype=DataType.VARCHAR, max_length=200), 
    FieldSchema(name='weatherid', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='weathername', dtype=DataType.VARCHAR, max_length=200), 
    FieldSchema(name='observationdate', dtype=DataType.VARCHAR, max_length=200), 
    FieldSchema(name='temperature', dtype=DataType.FLOAT), 
    FieldSchema(name='dewpoint', dtype=DataType.VARCHAR, max_length=200), 
    FieldSchema(name='relativehumidity', dtype=DataType.VARCHAR, max_length=200), 
    FieldSchema(name='windspeed', dtype=DataType.VARCHAR, max_length=200), 
    FieldSchema(name='winddirection', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='gust', dtype=DataType.VARCHAR, max_length=200), 
    FieldSchema(name='weather', dtype=DataType.VARCHAR, max_length=200), 
    FieldSchema(name='visibility', dtype=DataType.VARCHAR, max_length=200), 
    FieldSchema(name='altimeter', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='slp', dtype=DataType.VARCHAR, max_length=200), 
    FieldSchema(name='timezone', dtype=DataType.VARCHAR, max_length=200), 
    FieldSchema(name='state', dtype=DataType.VARCHAR, max_length=200), 
    FieldSchema(name='windchill', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='weatherdetails', dtype=DataType.VARCHAR, max_length=8000),    
    FieldSchema(name='image_vector', dtype=DataType.FLOAT_VECTOR, dim=512),
    FieldSchema(name='weather_text_vector', dtype=DataType.FLOAT_VECTOR, dim=384)
]

schema = CollectionSchema(fields=fields)

if milvus_client.has_collection(collection_name=COLLECTION_NAME):
    print("Exists.")
else:
    milvus_client.create_collection(COLLECTION_NAME, schema=schema, metric_type="COSINE", auto_id=True)

    index_params = milvus_client.prepare_index_params()
    index_params.add_index(field_name = "image_vector", metric_type="COSINE")
    
    index_params.add_index(
        field_name="id",
        index_type="STL_SORT"
    )
    
    index_params.add_index(
        field_name="weather_text_vector",
        index_type="IVF_FLAT",
        metric_type="L2",
        params={"nlist": 100}
    )
    
    milvus_client.create_index(COLLECTION_NAME, index_params)
    res = milvus_client.get_load_state(
        collection_name = COLLECTION_NAME
    )
    print(res)

# -----------------------------------------------------------------------------
# Iterate json urls
for jsonitems in json_object:
    if (  not jsonitems['Disabled'] and not jsonitems['Blocked'] ):
        latitude = jsonitems['Latitude']
        longitude = jsonitems['Longitude']
        weatherurl = "https://forecast.weather.gov/MapClick.php?lat={0}&lon={1}&unit=0&lg=english&FcstType=json".format(str(latitude), str(longitude))
        weatherfields = weatherparse(weatherurl)
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
        weatherdetails = 'The current weather observation for {0}[{1}] in {2} @ {3},{4} for {5} is {6} with a temperature of {7}F, a dew point of {8} and relative humidity of {9}% with a wind speed of {10}{11} with a visibility of {12} at an elevation of {13} and an altimeter reading of {14} for the {15} area.'.format(weatherfields['weathername'], weatherfields['weatherid'], weatherfields['state'], latitude, longitude, weatherfields['observationdate'], weatherfields['weather'], weatherfields['temperature'], weatherfields['dewpoint'], weatherfields['relativehumidity'], weatherfields['windspeed'], weatherfields['winddirection'], weatherfields['visibility'], weatherfields['elevation'], weatherfields['altimeter'], weatherfields['areadescription'])
        print(weatherdetails)
        filepath = "camimages/" + strfilename
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
            resultfilename = "camimages/yolo{0}.png".format(uuid.uuid4())
            result.save(filename=resultfilename)  # save to disk
            strText = ":tada:" + str(strname) + ":" + str(roadwayname)

# -----------------------------------------------------------------------------
# Slack
            SENDSLACK = False
            if ( SENDSLACK ): 
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
                        initial_comment="Transformed image " + str(resultfilename),
                    )
                except SlackApiError as e:
                    assert e.response["error"]

# -----------------------------------------------------------------------------
# Milvus insert
            try:
                imageembedding = extractor(resultfilename)
                weatherembedding = model(weatherdetails)
                milvus_client.insert( COLLECTION_NAME, {"image_vector": imageembedding, "weather_text_vector": weatherembedding, "filepath": resultfilename, 
                                                        "url": url,  "videourl": videourl, "latlong": latlong, "name": strname, 
                                                        "roadwayname": roadwayname, "directionoftravel": directionoftravel, "videourl": videourl, 
                                                        "creationdate": str(weatherfields['creationdate']), "areadescription": str(weatherfields['areadescription']),
                "elevation": str(weatherfields['elevation'] ), "county": str(weatherfields['county'] ),
                "metar": str(weatherfields['metar']), "weatherid": str(weatherfields['weatherid']), "weathername": str(weatherfields['weathername']),
                "observationdate": str(weatherfields['observationdate']), "temperature": float(weatherfields['temperature'] ),
                "dewpoint": str(weatherfields['dewpoint']) , "relativehumidity": str(weatherfields['relativehumidity']) ,
                "windspeed": str(weatherfields['windspeed']), "winddirection": str(weatherfields['winddirection']),"gust": str(weatherfields['gust'] ),
                "weather": str(weatherfields['weather']), "visibility": str(weatherfields['visibility']), 
                "altimeter": str(weatherfields['altimeter']), "slp": str(weatherfields['slp']), "timezone": str(weatherfields['timezone']), 
                "state": str(weatherfields['state']) ,"windchill": str(weatherfields['windchill']),"weatherdetails": str(weatherdetails) })
                        
                print("resultfilename:" + resultfilename)
                print("Milvus:sent collection:" + roadwayname)

            except Exception as e:
                print("An error:", e)

# -----------------------------------------------------------------------------
