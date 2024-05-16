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

#shutil.rmtree('runs/detect')

url = os.environ["NYURL"] 

response = requests.get(url).content

# print(response)
json_object = json.loads(response)

latitude = ""
longitude = ""
strid = ""
strname = ""
directionoftravel = ""
url = ""
videourl = ""
roadwayname = ""

model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model
slack_token = os.environ["SLACK_BOT_TOKEN"]
client = WebClient(token=slack_token)

for jsonitems in json_object:
    # now song is a dictionary
    if ( jsonitems['Url'] != '' and jsonitems['Disabled'] != 'True' and jsonitems['Blocked'] != 'True'):
        latitude = jsonitems['Latitude']
        longitude = jsonitems['Longitude']
        strid = jsonitems['ID']
        strname = jsonitems['Name']
        directionoftravel = jsonitems['DirectionOfTravel']
        roadwayname = jsonitems['RoadwayName']
        url = jsonitems['Url']
        videourl = jsonitems['VideoUrl']
        uuid2 = "{0}_{1}".format(strftime("%Y%m%d%H%M%S",gmtime()),uuid.uuid4())
        print ("URL:" + str(url))
        url = str(url) + "#" + str(uuid2)
        img = requests.get(url)
        strfilename = str(uuid2) + ".png"
        filepath = "/opt/demo/nifi-2.0.0-M2/camimages/" + strfilename
        if img.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(img.content)

        results = model.predict(filepath, stream=False, save=True, imgsz=640, conf=0.5)
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
            # result.save(filename=strfilename)  # save to disk
            # jsonoutput = result.tojson()

            print("outputimage:" + outputimage)
            print("names:" + json.dumps(names))
            print("speed:" + json.dumps(speed))
            strText = ":tada:" + str(strname) + ":" + str(roadwayname)

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
                    initial_comment="Transformed image " + str(strfilename),
                )
            except SlackApiError as e:
                # You will get a SlackApiError if "ok" is False
                assert e.response["error"]    # str like 'invalid_auth', 'channel_not_found'

            print("####")
        ### vectorize image
        ### send to milvus with metadata
       ##  print("Milvus attributes:" + strname )
#for r in results:
#	print(r.tojson())
