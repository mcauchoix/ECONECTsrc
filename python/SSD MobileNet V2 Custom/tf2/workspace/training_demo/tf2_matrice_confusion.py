# -*- coding: utf-8 -*-
"""
Pour lancer ce programme d'inférence sur un shell Anaconda :

conda activate tf2
cd C:\tf2\workspace\training_demo
python tf2_matrice_confusion.py
"""

import io
import os
import scipy.misc
import numpy as np
import six
import time
import glob
from IPython.display import display
import random

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, accuracy_score, average_precision_score
import seaborn as sns

from six import BytesIO

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

import tkinter

# Ignore Tensorflow WARNING and INFO
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Patch the location of gfile
tf.gfile = tf.io.gfile

# Ajouter une paire key:value à un dictionnaire
class dictionnary(dict): 
    # __init__ function 
    def __init__(self): 
        self = dict() 
          
    # Function to add key:value 
    def add(self, key, value): 
        self[key] = value 

# Comptage des espèces
def count_images_by_species():
  bird_species_train = dictionnary()
  bird_species_test = dictionnary()
  
  for folder in ['train','test']:
    # Récupération des fichiers CSV pour train et test
    CSV_path = '/tmpdir/DONNEEST21001/tf2/workspace/training_demo/annotations/' + folder + '.csv'
    df = pd.read_csv(CSV_path)
    class_count = df["class"].value_counts()
    print("Pour le " + folder + " :\n", class_count)

    # Récupérer toutes les images par espèce : MESCHA, ... etc 
    # TODO : et après avoir les sexes aussi => ajouter sur le xml_to_csv.py ???
    for espece in labels:
      all_rows = df.loc[df['class'] == espece]  # toutes les lignes qui correspondent à l'espèce courante
      # on garde seulement les noms des images associées => 'filename'
      if folder == "train":
        bird_species_train.add(espece, all_rows['filename'])
      else: 
        # Ajout dans la liste de test
        bird_species_test.add(espece, all_rows['filename'])
  return bird_species_train, bird_species_test

def load_groundtruths_on_Test(filename):
  # Récupération des fichiers CSV pour train et test
  CSV_path = '/tmpdir/DONNEEST21001/tf2/workspace/training_demo/annotations/test.csv'
  df = pd.read_csv(CSV_path)
  all_rows = df.loc[df['filename'] == filename]  # toutes les lignes qui correspondent à l'image courante

  # Parcours du nombre de boxes annotées
  GT = []
  for i in range(len(all_rows)):
    xmin = all_rows['xmin'].values[i]
    ymin = all_rows['ymin'].values[i]
    xmax = all_rows['xmax'].values[i]
    ymax = all_rows['ymax'].values[i]
    species = all_rows['class'].values[i]  # espèce annotée
    GT.append([ymin, xmin, ymax, xmax, species])  # ordre des arguments pour draw_bounding_box_on_image_array()

  return GT

def load_image_into_numpy_array(image): #adopted from object_detection_tutorial.ipynb
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(model, image, elapsed_time):
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  # Temps moyen de detection et classif
  start_time = time.time()
  output_dict = model_fn(input_tensor)
  elapsed_time.append(time.time() - start_time)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int32)

  # Récupère les coordonnées des boxes prédites : https://stackoverflow.com/questions/48915003/get-the-bounding-box-coordinates-in-the-tensorflow-object-detection-api-tutorial
  boxes = output_dict['detection_boxes'][0]
  # get all boxes from an array
  max_boxes_to_draw = boxes.shape[0]
  # get scores to get a threshold
  scores = output_dict['detection_scores']
  # this is set as a default but feel free to adjust it to your needs
  min_score_thresh=.5
  # iterate over all objects found
  coordinates = []
  predicted_classes = []
  for i in range(min(max_boxes_to_draw, boxes.shape[0])):
      if scores is None or scores[i] > min_score_thresh:
          class_name = category_index[output_dict['detection_classes'][i]]['name']
          predicted_classes.append(class_name)
          coordinates.append({
              "box": boxes,  # TODO construire liste JSON des prédictions avec le nom des images pour calculer les métriques : https://gist.github.com/tarlen5/008809c3decf19313de216b9208f3734
                             # + Groundtruths JSON aussi ==> Ligne 217 var. boxes
              "class_name": class_name,
              "score": scores[i]
          })

    #print(coordinates)  # TODO JSON pour calculer les métriques ???

  return output_dict, elapsed_time, predicted_classes

# Construit le fichier CSV d'évaluation par espèce
def construct_evaluation_CSV(data, image_path):
  # Pour chaque espèce on a une nouvelle ligne dans le CSV : liste d'images 
  # Pour chaque image on a le comptage des espèces annotées et détectées

  # TODO : parcourir les labels et créer le dictionnaire avec une boucle pour l'adapter au nombre d'espèces
  dict = {'ImageName': [], 'MESCHA_A' : 0, 'ECUROU_A' : 0,  'VEREUR_A' : 0, 'MESBLE_A' : 0, 'PIEBAV_A' : 0, 'MESNON_A' : 0, 'SITTOR_A' : 0, 'ACCMOU_A' : 0, 'ROUGOR_A' : 0, 'TOUTUR_A' : 0, 'PINARB_A' : 0, 'MOIDOM_A' : 0, 'MULGRI_A':0, 'CAMPAG_A':0, 
      	  'MESCHA_CV' : 0,  'ECUROU_CV' : 0,  'VEREUR_CV' : 0,  'MESBLE_CV' : 0,  'PIEBAV_CV' : 0,  'MESNON_CV' : 0,  'SITTOR_CV' : 0,  'ACCMOU_CV' : 0,  'ROUGOR_CV' : 0,  'TOUTUR_CV' : 0,  'PINARB_CV' : 0,  'MOIDOM_CV' : 0, 'MULGRI_CV':0, 'CAMPAG_CV':0,}

  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image = Image.open(image_path)
  image_np = load_image_into_numpy_array(image)

  # Actual detection.
  output_dict, elapsed_time, predicted_classes = run_inference_for_single_image(model, image_np, [])

  # Load groundtruths : on ne garde que le nom de l'image courante à partir du chemin
  image_name = os.path.basename(image_path)
  groundtruths = load_groundtruths_on_Test(image_name)
  for gt in groundtruths:
    _, _, _, _, species = gt

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

#########################################################################
######################### Partie  pour les tests #########################
#########################################################################
#########################################################################
# Chemins vers nos fichiers
#########################################################################
current_dir = os.getcwd() + '/workspace/training_demo/'  # modifier le chemin en fonction de l'emplacement du script SLURM (tf2_inference.slurm)
test_images_dir = current_dir + '/images/test/'
train_record_path = current_dir + '/annotations/train.record'
test_record_path = current_dir + '/annotations/test.record'

#labelmap_name = 'labelmap_12especes.pbtxt' # si on a les MESCHA : TrainV2 à V4 et TrainV6
#labelmap_name = 'labelmap_sansMESCHA.pbtxt' # TrainV5 => TODO param  of program !
labelmap_name = 'labelmap_14especes.pbtxt'
labelmap_path = current_dir + '/annotations/' + labelmap_name  # la labelmap d'entrainement sous forme d'objets

#########################################################################
# Récupération des labels pour nos classes
#########################################################################
# List of the strings that is used to add correct label for each box.
category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)

# Récupère la labelmap (liste des labels) pour les afficher sur légende sur des graphiques
labelmap =  label_map_util.create_categories_from_labelmap(labelmap_path, use_display_name=True)

# Récupération des labels pour les afficher sur la légende des graphiques
labels = [i['name'] for i in labelmap]

# Construction de la liste des labels associés avec leurs couleurs respectives
handles = []
# Couleurs utilisées dans utils https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py#:~:text=_TITLE_TOP_MARGIN%20%3D%2010-,STANDARD_COLORS%20%3D%20%5B,-%27AliceBlue%27%2C%20%27Chartreuse%27%2C%20%27Aqua
for i in range(len(labels)):
  patch = mpatches.Patch(color=viz_utils.STANDARD_COLORS[i+1], label=labels[i])  # i+1 pour les couleurs car l'index de la labelmap commence à 1 (et la boucle à 0)
  handles.append(patch)

# Ajoute la couleur rouge pour les labels/annotations/groundtruths :
handles.append(mpatches.Patch(color='red', label='Labels'))

#########################################################################
# Récupération de notre modèle CUSTOM
#########################################################################
tf.keras.backend.clear_session()
# TODO param of program !
#exported_model_name = 'TF2_ExportedModel_TrainV8'
#exported_model_name = 'SSD_320x320_trainV2'
#exported_model_name = 'SSD_8'
exported_model_name = 'SSD_64'
exported_model_folder = "/exported-models/" + exported_model_name + "/saved_model"
model = tf.saved_model.load(current_dir + exported_model_folder)

#########################################################################
# Récupération de toutes les images d'une classe donnée, et inférence sur
# N images de chaque espèce !
# Remarque : Appuyer sur la touche 'Q' ou fermer la fenêtre d'affichage
# pour passer à l'image suivante
#########################################################################
# TODO : Adapter le nom du CSV généré !
CSV_path = '/tmpdir/DONNEEST21001/tf2/workspace/training_demo/_evalCSV/' + 'Evaluation-SSD_batch8.csv'

# TODO : passer en argument du programme les paramètres
#N_images = 2
N_images = len(os.listdir(test_images_dir))  # si on veut le faire sur toutes les images de test

# Comptage des espèces annotées et détectées 
data = pd.DataFrame()

cpt = 0
images_list = glob.glob(test_images_dir + '*.jpg')
for img_path in images_list:
  if (cpt % 100) == 0:
    print('Progression : ', cpt, ' sur : ', len(images_list))
  data = construct_evaluation_CSV(data, img_path)
  cpt += 1

# Construit le fichier CSV d'évaluation
print('Saving CSV for evaluation ...')
data.to_csv(CSV_path, sep=';', index=None)

# Matrice de confusion à partir des CSV

# Lecture du fichier CSV avec ';' comme délimiteur/séparateur
# TODO : Adapter le CSV path (ligne 351)
test_df = pd.read_csv(CSV_path, sep=';')

mean_ACC = []
mean_precision = []
mean_recall = []
for species in labels:
  print('Espèce courante : ', species)
  actual_df = test_df[species + '_A']
  predicted_df = test_df[species + '_CV']

  # Ne garde que la liste des valeurs
  actual_list = actual_df.values
  predicted_list = predicted_df.values
  #print('values count : ', len(actual_list))

  # Construction de la matrice de confusion
  # On transforme la liste des vérités terrain en SET pour avoir l'ensemble des valeurs uniques
  unique_values = list(set(actual_list) | set(predicted_list))  # exemple : [1.0, 2.0, 3.0] si on a de 1 à 3 oiseaux de la même espèce sur l'image

  # On supprime les valeurs à 0 de la colonne pour l'espèce courante (ce sont les lignes qui sont d'une autre espèce)
  unique_values = unique_values[1:]

  # Calcul de la matrice de confusion
  try:
    cf_matrix = confusion_matrix(actual_list, predicted_list, labels=unique_values)
    print('Confusion matrix : \n', cf_matrix)

    # classification report for precision, recall f1-score and accuracy
    matrix = classification_report(actual_list, predicted_list, labels=unique_values, zero_division = 0)
    print('Classification report : \n',matrix)

    # Calcul manuelle de l'Accuracy par classe
    FP = cf_matrix.sum(axis=0) - np.diag(cf_matrix)  
    FN = cf_matrix.sum(axis=1) - np.diag(cf_matrix)
    TP = np.diag(cf_matrix)
    TN = cf_matrix.sum() - (FP + FN + TP)
    print('TP, FP, TN, FN = ', TP, FP, TN, FN)

    # Overall accuracy
    numerateur = (TP+TN)
    denumerateur = (TP+FP+FN+TN)
    if np.sum(denumerateur) != 0:
      ACC = numerateur / denumerateur
      mean_ACC.append(np.mean(ACC))
    else:
      # Si on n'a aucune détection pour une classe, on a une division par zéro donc on fixe la précision à 0 pour cette classe
      ACC = 0.0
      mean_ACC.append(ACC)

    accuracy = accuracy_score(actual_list, predicted_list)
    precision = precision_score(actual_list, predicted_list, labels=unique_values, average='micro')
    mean_precision.append(precision)
    recall = recall_score(actual_list, predicted_list, labels=unique_values, average='micro')
    mean_recall.append(precision)
    print('Accuracy (cf. formule) : ', accuracy)
    print('Precision: %.3f' % precision)
    print('Recall: %.3f' % recall)
  except ValueError:
    print("Pas de d'images de TEST dans ce split pour l'espèce courante !")

print('ACCURACY MOYENNE : ', np.mean(mean_ACC))
print('PRECISION MOYENNE : ', np.mean(mean_precision))
print('RECALL MOYENNE : ', np.mean(mean_recall))