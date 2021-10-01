# coding: utf-8
# This is Edje Electronics' code with some minor adjustments
# Credit goes to his repo: https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi

import tflite_runtime.interpreter as tflite
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
from datetime import datetime
from datetime import timedelta
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Provide the path to the TFLite file, default is models/model.tflite',
                    default='models/model.tflite')
parser.add_argument('--labels', help='Provide the path to the Labels, default is models/labels.txt',
                    default='models/labels.txt')
parser.add_argument('--video', help='Name of the video to perform detection on',
                    default='videos/2021-09-16-09-18-27.mp4')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.65)
parser.add_argument('--keepDetections', help='Saves detections (boxes, classes and scores) on frames',
                    default=False)
                    
args = parser.parse_args()

# PROVIDE PATH TO MODEL DIRECTORY
PATH_TO_MODEL_DIR = args.model

# PROVIDE PATH TO LABEL MAP
PATH_TO_LABELS = args.labels

# PROVIDE PATH TO VIDEO DIRECTORY
VIDEO_PATH = args.video

# PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
MIN_CONF_THRESH = float(args.threshold)

# Save detections on frames or not
keepDetections = args.keepDetections

import time
print('Loading model...', end='')
start_time = time.time()

# LOAD TFLITE MODEL
interpreter = tflite.Interpreter(model_path=PATH_TO_MODEL_DIR)
# LOAD LABELS
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
print('Running inference for {}... '.format(VIDEO_PATH), end='')
# Initialize video
video = cv2.VideoCapture(VIDEO_PATH)
imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Variables propres à la vidéo
video_name = os.path.basename(VIDEO_PATH)
fps = video.get(cv2.CAP_PROP_FPS)
number_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
print('\nNombre d\'images par secondes (FPS) : ', fps)
print('Total number of frames : ', number_of_frames)
print('Durée de la vidéo : ', number_of_frames/fps, ' secondes')

# Dossier pour sauvegarder les frames (le créé s'il n'existe pas) :
frame_path = os.getcwd() + '/saved_frames/'
if keepDetections and not os.path.exists(frame_path):
  os.mkdir(frame_path)

# Les frames sont sauvegardées dans un dossier qui a le même nom que la vidéo courante
frame_path += video_name
if not os.path.exists(frame_path):
  os.mkdir(frame_path)

# Converti la liste de labels en dictionnaire pour compter les occurences des espèces
counting_detections = dict.fromkeys(labels, 0)

# Garde les N frames denières détections pour supprimer les faux positifs
N_save = 5 # Par défaut, si on veut que l'oiseau y soit au moins 1 seconde, on fixe cette valeur à la variable "fps"

# Pour le fichier CSV, qui a pour nom : CSV_nomVidéo (sans .mp4)
CSV_path = os.getcwd() + '/CSV_' + video_name[:-4] + '.csv'
print('CSV_path :', CSV_path)
header = ['videotime', 'class', 'xmin', 'ymin', 'xmax', 'ymax']

# Créé le fichier CSV en écriture
with open(CSV_path, 'w', newline='') as f:
    # Entête du fichier CSV seulement à la création du fichier
    writer = csv.DictWriter(f, delimiter=';', fieldnames=header)
    writer.writeheader()

# Parcours de la vidéo frame par frame
while(video.isOpened()):
    # Start timer (for calculating frame rate)
    current_count=0
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    ret, frame1 = video.read()

    # Si on a bien lu une frame
    if ret:
        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)
        
        # Sauvegarde le temps de la frame courante en secondes
        milliseconds = video.get(cv2.CAP_PROP_POS_MSEC)
        timestamp = timedelta(milliseconds=milliseconds)
        
        # Numéro de la frame courante
        frame_number = video.get(cv2.CAP_PROP_POS_FRAMES)
        if frame_number % 50 == 0:
            print('Treating frame n°', frame_number)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
        #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        detected_classes = []
        detected_scores = []
        keep_frame = True
        
        # Pour le fichier CSV
        rows = []
        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > MIN_CONF_THRESH) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                # Trace la bounding box sur l'image
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                score = int(scores[i]*100)
                label = '%s: %d%%' % (object_name, score) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                current_count+=1

                # Mets à jour le compteur de la classe actuelle dans le dictionnaire
                counting_detections[object_name] += 1
                # Enregistre les noms de classes et scores détectés
                detected_classes.append(object_name)
                detected_scores.append(scores[i])
                # Vérifie si l'espèce détectée n'est pas présente sur au moins N frames, on ne la garde pas
                if counting_detections[object_name] < N_save :
                    keep_frame = False
                    
                # Données pour la/les nouvelle(s) ligne(s) pour le CSV
                rows.append({'videotime': timestamp, 'class': object_name, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})

        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1

        # Draw framerate in corner of frame
        cv2.putText(frame,'FPS on Pi: {0:.2f}, Detection(s) count: {1}'.format(frame_rate_calc, str(current_count)),(15,25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,55),2,cv2.LINE_AA)
        # Si on a détecté une espèce au moins : 
        if detected_classes and keep_frame :
          # Sauvegarde la frame avec les détections et le temps sur la vidéo (si ce temps n'est pas nul)
          # TODO les 5 ou 6 dernières frames ont un temps égal à 0.0 avec cv2 ???
          if milliseconds != 0.0 and keepDetections:
            # temps de la frame ==> str(timestamp)
            saved_frame_name = 'frame_' + str(int(frame_number)) + '.jpg'
            cv2.imwrite(os.path.join(frame_path, saved_frame_name), frame)
          elif not keepDetections:
            # Affiche les détections à la volée
            cv2.imshow('Object detector', frame)

        # Ouverture du fichier CSV pour ajouter des données à la suite
        with open(CSV_path, 'a', newline='') as f:
            # Sauvegarde le nom de l'image, la/les espèce(s) et les coordonées des bounding boxes dans le CSV
            writer = csv.DictWriter(f, delimiter=';', fieldnames=header)
            # Si on a au moins une détection
            if rows != []:
                # Une ligne par espèce pour l'image courante
                for l in rows:
                    writer.writerow(l)

        # Press 'q' to quit
        if not keepDetections:
            # Attend moins de temps pour sauvegarder les images/frames
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            # Affiche une image par seconde (1000ms)
            if cv2.waitKey(1000) == ord('q'):
                break
            
    # Si c'est la fin de la vidéo
    else:
        video.release()

# Affichage finaux
if os.path.exists(frame_path):
  print('Saved frames : ', len(os.listdir(frame_path)))

total_seconds = number_of_frames / frame_rate_calc
print('Average total time :', timedelta(seconds=total_seconds))
print('Detected species : ', counting_detections)

# Clean up
cv2.destroyAllWindows()
print("Done")