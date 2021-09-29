######## Evaluating Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Florent Gimenez
# Date: 06/09/21
# Description: 
# This program uses a TensorFlow Lite object detection model to perform object 
# detection on an image or a folder full of images.
# Then it evaluates detections per species according to groundtruths

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import glob
import importlib.util
import time
from xml.etree import ElementTree
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, accuracy_score, average_precision_score
import seaborn as sns

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='model.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--image', help='Name of the single image to perform detection on. To run detection on multiple images, use --imagedir',
                    default=None)
parser.add_argument('--imagedir', help='Name of the folder containing images to perform detection on. Folder must contain only images.',
                    default=None)
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
use_TPU = args.edgetpu

# Parse input image name and directory. 
IM_NAME = args.image
IM_DIR = args.imagedir

# If both an image AND a folder are specified, throw an error
if (IM_NAME and IM_DIR):
    print('Error! Please only use the --image argument or the --imagedir argument, not both. Issue "python TFLite_detection_image.py -h" for help.')
    sys.exit()

# If neither an image or a folder are specified, default to using 'test1.jpg' for image name
if (not IM_NAME and not IM_DIR):
    IM_NAME = 'test1.jpg'

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'

# Get path to current working directory
CWD_PATH = os.getcwd()

# Define path to images and grab all image filenames
if IM_DIR:
    PATH_TO_IMAGES = os.path.join(CWD_PATH,IM_DIR)
    images = sorted(glob.glob(PATH_TO_IMAGES + '/*.jpg'))

elif IM_NAME:
    PATH_TO_IMAGES = os.path.join(CWD_PATH,IM_NAME)
    images = glob.glob(PATH_TO_IMAGES)

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5
elapsed_time = []

# Plus rapide qu'avec pandas !
def get_labels_from_xml(image_path):
    xml_path = image_path.replace('.jpg', '.xml')
    # Parse XML
    tree = ElementTree.parse(xml_path)
    root = tree.getroot()
    
    # Get names and values of "species" fields
    names = root.findall('.//attribute/name')
    values = root.findall('.//attribute/value')
    indices = [i for i, x in enumerate(names) if x.text == 'species']
    
    return [values[i].text for i in indices]

# avec pandas depuis le CSV
"""
def load_labels_from_csv(image_path):
    #xml_path = image_path.replace('.jpg', '.xml')
    image_name = os.path.basename(image_path)
    CSV_path = 'annotations/test.csv'
    df = pd.read_csv(CSV_path)
    
    # Get all rows for current image/XML
    all_rows = df.loc[df['filename'] == image_name]
    
    # Returns groundtruths
    return [all_rows['class'].values[i] for i in range(len(all_rows))]
"""

def construct_evaluation_CSV(data, image_path, predicted_classes):
  # Pour chaque espèce on a une nouvelle ligne dans le CSV : liste d'images 
  # Pour chaque image on a le comptage des espèces annotées et détectées

  dict = {'ImageName': [], 'MESCHA_A' : 0, 'ECUROU_A' : 0,  'VEREUR_A' : 0, 'MESBLE_A' : 0, 'PIEBAV_A' : 0, 'MESNON_A' : 0, 'SITTOR_A' : 0, 'ACCMOU_A' : 0, 'ROUGOR_A' : 0, 'TOUTUR_A' : 0, 'PINARB_A' : 0, 'MOIDOM_A' : 0, 
      'MESCHA_CV' : 0,  'ECUROU_CV' : 0,  'VEREUR_CV' : 0,  'MESBLE_CV' : 0,  'PIEBAV_CV' : 0,  'MESNON_CV' : 0,  'SITTOR_CV' : 0,  'ACCMOU_CV' : 0,  'ROUGOR_CV' : 0,  'TOUTUR_CV' : 0,  'PINARB_CV' : 0,  'MOIDOM_CV' : 0}

  # On ne garde que le nom de l'image courante à partir du chemin
  image_name = os.path.basename(image_path)

  # Load labels from the XML file of the current image
  groundtruths = get_labels_from_xml(image_path)

  for species in groundtruths:
    # Ajoute 1 à l'espèce courante annotée
    column_name = species + '_A'
    dict[column_name] += 1

  # De même pour les classes prédites
  for c in predicted_classes:
    # Ajoute 1 à l'espèce courante détectée !
    column_name = c + '_CV'
    dict[column_name] += 1

  # Ajoute les nom de l'image
  dict['ImageName'] = image_name

  # Ajoute la ligne pour l'image courante
  data = data.append(dict, ignore_index=True, sort=False)
  return data

###########################################################################
cpt = 0
data = pd.DataFrame()

# Loop over every image and perform detection
# Images triées alphabétiquement
for image_path in images:
    # Affichage
    if (cpt % 100) == 0:
        print('Progression : ', cpt, ' sur : ', len(images))
    
    # Load image and resize to expected shape [1xHxWx3]
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape 
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    
    # Time needed for one detection
    start_time = time.time()
    interpreter.invoke()
    elapsed_time.append(time.time() - start_time)

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    predicted_classes = []
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            predicted_classes.append(object_name)

    # TODO : Construit le fichier CSV pour l'évaluation manuelle
    data = construct_evaluation_CSV(data, image_path, predicted_classes)
    cpt += 1

# Sauvegarde le dictionnaire final dans un fichier CSV
print('Saving CSV for evaluation ...')
CSV_path = os.path.join(CWD_PATH,'EvaluationSSD_V2_RaspberryPi.csv')  # TODO param prog ?
data.to_csv(CSV_path, sep=';', index=None)

# Prints mean elapsed time per detection
if elapsed_time != [] :
    mean_elapsed = sum(elapsed_time) / float(len(elapsed_time))
    print('Mean elapsed time : ', str(mean_elapsed), ' seconds per detection')