#!/usr/bin/env python
# coding: utf-8
"""
Object Detection (On Video) From TF2 Saved Model
=====================================
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import cv2
import argparse
from datetime import timedelta

import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings
tf.get_logger().setLevel('ERROR')   # Suppress TensorFlow logging (2)

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Folder that the Saved Model is Located In',
                    default='exported-models/my_mobilenet_model')
parser.add_argument('--labels', help='Where the Labelmap is Located',
                    default='exported-models/my_mobilenet_model/saved_model/label_map.pbtxt')
parser.add_argument('--video', help='Name of the video to perform detection on. To run detection on multiple images, use --imagedir',
                    default='test.mp4')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.65)

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.
    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.
    Args:
      path: the file path to the image
    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Browses program's arguments
args = parser.parse_args()

# PROVIDE PATH TO IMAGE DIRECTORY
VIDEO_PATHS = args.video

# PROVIDE PATH TO MODEL DIRECTORY
PATH_TO_MODEL_DIR = args.model

# PROVIDE PATH TO LABEL MAP
PATH_TO_LABELS = args.labels

# PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
MIN_CONF_THRESH = float(args.threshold)

# Load the model
# ~~~~~~~~~~~~~~
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

# Mesure le temps pour charger le mod??le
print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# Load label map data (for plotting)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

# R??cup??re la labelmap (liste des labels) pour les afficher sur l??gende sur des graphiques
labelmap =  label_map_util.create_categories_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# R??cup??ration des labels pour les afficher sur la l??gende des graphiques
labels = [i['name'] for i in labelmap]
print('Running inference for {}... '.format(VIDEO_PATHS), end='')

# Variables propres ?? la vid??o
video_name = os.path.basename(VIDEO_PATHS)
video = cv2.VideoCapture(VIDEO_PATHS)
fps = video.get(cv2.CAP_PROP_FPS)
number_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
print('\nNombre d\'images par secondes (FPS) : ', fps)
print('Total number of frames : ', number_of_frames)
print('Dur??e de la vid??o : ', number_of_frames/fps, ' secondes')

# Dossier pour sauvegarder les frames (le cr???? s'il n'existe pas) :
frame_path = os.getcwd() + '\\saved_frames\\'
if not os.path.exists(frame_path):
  os.mkdir(frame_path)

# Les frames sont sauvegard??es dans un dossier qui a le m??me nom que la vid??o courante
frame_path += video_name
if not os.path.exists(frame_path):
  os.mkdir(frame_path)

# Converti la liste de labels en dictionnaire pour compter les occurences d'esp??ces
counting_detections = dict.fromkeys(labels, 0)

# Garde les N frames deni??res d??tections pour supprimer les faux positifs
N_save = 5  # Par d??faut, si on veut que l'oiseau y soit au moins 1 seconde, on fixe cette valeur ?? la variable "fps"

# Parcours la vid??o frame par frame
while(video.isOpened()):
    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()

    # Si on a bien lu une frame
    if ret:
      frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
      frame_expanded = np.expand_dims(frame_rgb, axis=0)

      # Num??ro de la frame courante
      frame_number = video.get(cv2.CAP_PROP_POS_FRAMES)

      # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
      input_tensor = tf.convert_to_tensor(frame)
      # The model expects a batch of images, so add an axis with `tf.newaxis`.
      input_tensor = input_tensor[tf.newaxis, ...]

      # Run inference
      model_fn = detect_fn.signatures['serving_default']
      detections = model_fn(input_tensor)

      # All outputs are batches tensors.
      # Convert to numpy arrays, and take index [0] to remove the batch dimension.
      # We're only interested in the first num_detections.
      num_detections = int(detections.pop('num_detections'))
      detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
      detections['num_detections'] = num_detections

      # detection_classes should be ints.
      detections['detection_classes'] = detections['detection_classes'].astype(np.int32)

      # R??cup??re les classes d??tect??es :
      # R??cup??re les coordonn??es des boxes pr??dites
      boxes = detections['detection_boxes'][0]
      # get all boxes from an array
      max_boxes_to_draw = boxes.shape[0]
      # get scores to get a threshold
      scores = detections['detection_scores']
      # iterate over all objects found
      coordinates = []
      detected_classes = []
      detected_scores = []
      keep_frame = True
      for i in range(max_boxes_to_draw):
          if scores is None or scores[i] > MIN_CONF_THRESH:
              class_name = category_index[detections['detection_classes'][i]]['name']
              # Mets ?? jour le compteur de la classe actuelle dans le dictionnaire
              counting_detections[class_name] += 1
              # Enregistre les noms de classes et scores d??tect??s
              detected_classes.append(class_name)
              detected_scores.append(scores[i])
              # V??rifie si l'esp??ce d??tect??e n'est pas pr??sente sur au moins N frames, on ne la garde pas
              if counting_detections[class_name] < N_save :
                  keep_frame = False

      frame_with_detections = frame.copy()
      
      # SET MIN SCORE THRESH TO MINIMUM THRESHOLD FOR DETECTIONS
      viz_utils.visualize_boxes_and_labels_on_image_array(
            frame_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=MIN_CONF_THRESH,
            agnostic_mode=False)

      # Affichage des d??tections sur la vid??o
      #cv2.imshow('Object Detector', frame_with_detections)

      # Si on a d??tect?? une esp??ce au moins : 
      if detected_classes and keep_frame :
        # Sauvegarde le temps de la frame courante en secondes
        milliseconds = video.get(cv2.CAP_PROP_POS_MSEC)
        timestamp = timedelta(milliseconds=milliseconds)
        # Sauvegarde la frame avec les d??tections et le temps sur la vid??o (si ce temps n'est pas nul)
        # TODO les 5 ou 6 derni??res frames ont un temps ??gal ?? 0.0 avec cv2 ???
        if milliseconds != 0.0:
          # temps de la frame ==> str(timestamp)
          saved_frame_name = 'frame_num' + str(int(frame_number)) + '.jpg'
          cv2.imwrite(os.path.join(frame_path, saved_frame_name), frame_with_detections)

      if cv2.waitKey(1) == ord('q'):
          break
    # Apr??s avoir examin?? toutes les frames, termine la boucle
    else:
      video.release()

print('Saved frames : ', len(os.listdir(frame_path)))
print('Detected species : ', counting_detections)
cv2.destroyAllWindows()