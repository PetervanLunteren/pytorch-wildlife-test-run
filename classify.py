# This script to test if AI4G models can be deployed using the EcoAssist workflow
# Below you can find a minimal reproducable example
# You can test it yourself by cloning the repo
# Feel free to PR with adjustments

# EcoAssist works with the following workflow to classify animals:
    # Step 1: loop through the MegaDetector output JSON to find where the animals are
    # Step 2: crop out each animal and feed it to the classifier
    # Step 3: add its predictions to the existing JSON

# Execute script:
    # conda activate pytorch-wildlife && python "C:\Users\smart\Desktop\pytorch-wildlife-test-run\classify.py"

# set working directory to file location
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

from PytorchWildlife.models import classification as pw_classification
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import json
import supervision as sv
import numpy as np

# load classifier from local file
try:
    classification_model = pw_classification.AI4GAmazonRainforest(weights = "AI4GAmazonClassification_v0.0.0.ckpt")
except FileNotFoundError:
    print(f"Please download the model file from the url below and place it in the dir {dname}.\n")
    print("https://zenodo.org/records/10042023/files/AI4GAmazonClassification_v0.0.0.ckpt?download=1")
    exit()

# method of cropping
# for the ecoassist workflow we need to define a function that crops and pads the images from the bounding 
# box coordinates to have the animals be presented exactly like the ones the classifier was trained on
# input: image = full image PIL.Image.open(img_fpath) <class 'PIL.JpegImagePlugin.JpegImageFile'>
# input: bbox = the bbox coordinates as read from the MD json - detection['bbox'] - [xmin, ymin, xmax, ymax]
# output: cropped image <class 'PIL.Image.Image'>
# each developer has its own way of padding, squaring, cropping, resizing etc
# it eneeds to happen exactly th same as on which the model was trained
# ISSUE: the function below is just an example - could you share the code which was used to crop the animals during training?
def get_crop(img, bbox_norm):

    # convert bbox to int
    img_width, img_height = img.size
    left = int(round(bbox_norm[0] * img_width))
    top = int(round(bbox_norm[1] * img_height))
    right = int(round(bbox_norm[2] * img_width)) + left
    bottom = int(round(bbox_norm[3] * img_height)) + top

    # crop using supervision method
    crop = Image.fromarray(sv.crop_image(np.array(img.convert("RGB")), xyxy = [left, top, right, bottom]))
    return crop

# predict from cropped image
# for the ecoassist workflow we need to define a function that takes a cropped image and returns a list of all classes with their associated predictions
# input: cropped PIL image
# output: unsorted classifications formatted as [['aardwolf', 2.3025326090220233e-09], ['african wild cat', 5.658252888451898e-08], ... ]
# ISSUE: The code below will provide only the class with highest confidence. Is it possible to get all classes and their associated confidences? Something like:
# [['Dasyprocta', 0.001], ['Bos', 0.002], ['Pecari', 0.0005], ['Mazama', 0.746], ['Cuniculus', 0.023], ['Leptotila', 0.0045], ...]
def get_prediction(PIL_crop):
    transform = transforms.ToTensor()
    tensor = transform(PIL_crop)
    classification_results = classification_model.single_image_classification(tensor)
    print(classification_results)

# exmaple workflow
# loop json, crop image, get prediction
with open("imgs\image_recognition_file.json") as image_recognition_file_content:
    data = json.load(image_recognition_file_content)
    for image in data['images']:
        fname = image['file']
        for detection in image['detections']:
            category_id = detection['category']
            if category_id == '1':
                img_fpath = os.path.join(dname, fname)
                bbox = detection['bbox']
                crop = get_crop(Image.open(img_fpath), bbox)
                get_prediction(crop)
                
