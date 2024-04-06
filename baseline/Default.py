import random
import math
import sys
import hashlib
import numpy
import copy
import pandas as pd
from deap import creator, base, tools, algorithms
import torch
from scipy.spatial.distance import cosine
import requests
from PIL import Image
from statistics import mean
from ultralytics import YOLO
import time

import logging
from bs4 import BeautifulSoup
import requests
import schedule
from diffusers import (
    StableDiffusionPipeline,
    EulerDiscreteScheduler,
    StableDiffusionImg2ImgPipeline,
)
import calendar
import time
import numpy as np
from io import BytesIO
import cv2
import random
import argparse


# Optimizer parameters
# numTuples = int(ConfigSectionMap("Optimizer")['numtuples'])


def int_to_binary_and_select_elements(integer, element_list):
    binary_representation = bin(integer)[2:]
    selected_elements = []
    for i, digit in enumerate(binary_representation):
        if digit == "1":
            selected_elements.append(element_list[i])
    return selected_elements


# Parameters for the boxes
thickness = 2
fontScale = 0.5

model_id = "stabilityai/stable-diffusion-2"
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, scheduler=scheduler, torch_dtype=torch.float16
)  # for cuda
# pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float32) # for cpu
pipe = pipe.to("cuda")
# pipe = pipe.to("cpu")
model = YOLO("yolov8n.pt")  # load a pretrained YOLOv8n detection model
model.train(data="coco128.yaml", epochs=3)  # train the model
colors = np.random.randint(0, 255, size=(len(model.names), 3), dtype="uint8")

def read_box(box):
    cords = box.xyxy[0].tolist()
    cords = [round(x) for x in cords]
    class_id = model.names[box.cls[0].item()]
    conf = round(box.conf[0].item(), 2)
    return [class_id, cords, conf]


def addBoxesImage(currentImage, boxesInfo):
    image = cv2.imread(currentImage)
    for box in boxesInfo:
        class_id = box[0]
        confidence = box[2]
        color = [int(c) for c in colors[list(model.names.values()).index(class_id)]]
        #        color = colors[list(model.names.values()).index(class_id)]
        cv2.rectangle(
            image,
            (box[1][0], box[1][1]),
            (box[1][2], box[1][3]),
            color=color,
            thickness=thickness,
        )
        text = f"{class_id}: {confidence:.2f}"
        (text_width, text_height) = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, thickness=thickness
        )[0]
        text_offset_x = box[1][0]
        text_offset_y = box[1][1] - 5
        box_coords = (
            (text_offset_x, text_offset_y),
            (text_offset_x + text_width + 2, text_offset_y - text_height),
        )
        overlay = image.copy()
        cv2.rectangle(
            overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED
        )
        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
        cv2.putText(
            image,
            text,
            (box[1][0], box[1][1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=fontScale,
            color=(0, 0, 0),
            thickness=thickness,
        )
    cv2.imwrite(currentImage + "_yolo8.png", image)


def createNegativePrompt(selection):
    items = [
        "illustration",
        "painting",
        "drawing",
        "art",
        "sketch",
        "lowres",
        "error",
        "cropped",
        "worst quality",
        "low quality",
        "jpeg artifacts",
        "out of frame",
        "watermark",
        "signature",
    ]
    # integer_input =  random.randint(0,2**len(fixed_length_list)-1)
    if selection > 2 ** len(items) - 1:
        selection %= 2 ** len(items) - 1
    selected_elements = int_to_binary_and_select_elements(selection, items)
    return ", ".join(selected_elements)


def createPosPrompt(prompt, selection):
    items = [
        "photograph",
        "digital",
        "color",
        "Ultra Real",
        "film grain",
        "Kodak portra 800",
        "Depth of field 100mm",
        "overlapping compositions",
        "blended visuals",
        "trending on artstation",
        "award winning",
    ]
    # integer_input =  random.randint(0,2**len(fixed_length_list)-1)
    if selection > 2 ** len(items) - 1:
        selection %= 2 ** len(items) - 1
    selected_elements = int_to_binary_and_select_elements(selection, items)
    return prompt + ", " + ", ".join(selected_elements)


def text2img(prompt, configuration={}):
    # num_inference_steps = configuration["num_inference_steps"]
    # guidance_scale = configuration["guidance_scale"]
    # negative_prompt = createNegativePrompt(configuration["negative_prompt"])
    # prompt = createPosPrompt(prompt, configuration["positive_prompt"])
    # guidance_rescale = configuration["guidance_rescale"]
    # num_images_per_prompt = 4
    # seed = 0
    # generator = torch.Generator("cuda").manual_seed(seed)
    # generator = torch.Generator("cpu").manual_seed(seed)
    # print(prompt)
    # print(negative_prompt)

    #Measure GPU time
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter.record()
    imagesAll = pipe(
        prompt,
        # guidance_scale=guidance_scale,
        # num_inference_steps=num_inference_steps,
        # guidance_rescale=guidance_rescale,
        # negative_prompt=negative_prompt,
        # generator=generator,
        # num_images_per_prompt=num_images_per_prompt,
    ).images
    ender.record()
    torch.cuda.synchronize()
    inference_time = starter.elapsed_time(ender) / 60000 # compute inference time in minutes

    # print(inference_time)
    # print(imagesAll)
    timestamp = calendar.timegm(time.gmtime())
    images = []
    for i, image in enumerate(imagesAll):
        image.save(
            prompt.replace(" ", "_")
            + "."
            + str(timestamp)
            + "."
            + str(i)
            + "."
            + "image.png"
        )
        images.append(
            prompt.replace(" ", "_")
            + "."
            + str(timestamp)
            + "."
            + str(i)
            + "."
            + "image.png"
        )
    return images, inference_time


def img2text(image_path):
    result = model(image_path)  # predict on an image
    boxesInfo = []
    counting = {}
    for box in result[0].boxes:
        currentBox = read_box(box)
        boxesInfo.append(currentBox)
        if currentBox[0] in counting.keys():
            counting[currentBox[0]] += 1
        else:
            counting[currentBox[0]] = 1
    return counting, boxesInfo



prompt = "Two people and a bus"

avgPrecision = 0
totalCount = 0
# configuration = {
#     "num_inference_steps": individual["num_inference_steps"],
#     "guidance_scale": individual["guidance_scale"],
#     "negative_prompt": individual["negative_prompt"],
#     "positive_prompt": individual["positive_prompt"],
#     "guidance_rescale": individual["guidance_rescale"],
#     "seed": individual["seed"],
# }

# allimages, inference_time = text2img(self.prompt, configuration)
allimages, inference_time = text2img(prompt)
for currentImage in allimages:
    counting, boxesInfo = img2text(currentImage)
    print(counting)
    addBoxesImage(currentImage, boxesInfo)
    for box in boxesInfo:
        totalCount += 1
        avgPrecision += box[2]

if avgPrecision == 0:
    image_quality = 0
else:
    image_quality = avgPrecision / totalCount

print('image_quality: ', inference_time)
print('inference_time: ', inference_time)