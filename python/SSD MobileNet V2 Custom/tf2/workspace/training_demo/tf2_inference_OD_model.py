# -*- coding: utf-8 -*-
"""Tensorflow 2 Object Detection: Train model.ipynb

Original file is located at
    https://colab.research.google.com/github/TannerGilbert/Tensorflow-Object-Detection-API-Train-Model/blob/master/Tensorflow_2_Object_Detection_Train_model.ipynb

This notebook walks you through training a custom object detection model using the Tensorflow Object Detection API and Tensorflow 2.

The notebook is split into the following parts:
* Install the Tensorflow Object Detection API
* Prepare data for use with the OD API
* Write custom training configuration
* Train detector
* Export model inference graph
* Test trained model
"""
"""
batch_size = 16
num_steps = 8000
num_eval_steps = 1000
fine_tune_checkpoint = 'efficientdet_d0_coco17_tpu-32/checkpoint/ckpt-0'
base_config_path = 'ssd_efficientdet_d0_512x512_coco17_tpu-8.config'
# TODO automatiser la création du fichier labelmap.pbtxt : https://github.com/nicknochnack/TensorflowObjectDetectionMetrics/blob/main/Tutorial-Walkthrough.ipynb
# TODO automatiser modif config
# edit configuration file (from https://colab.research.google.com/drive/1sLqFKVV94wm-lglFq_0kGo2ciM0kecWD)
import re

with open(base_config_path) as f:
    config = f.read()

with open('model_config.config', 'w') as f:
  
  # Set labelmap path
  config = re.sub('label_map_path: ".*?"', 
             'label_map_path: "{}"'.format(labelmap_path), config)
  
  # Set fine_tune_checkpoint path
  config = re.sub('fine_tune_checkpoint: ".*?"',
                  'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), config)
  
  # Set train tf-record file path
  config = re.sub('(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")', 
                  'input_path: "{}"'.format(train_record_path), config)
  
  # Set test tf-record file path
  config = re.sub('(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")', 
                  'input_path: "{}"'.format(test_record_path), config)
  
  # Set number of classes.
  config = re.sub('num_classes: [0-9]+',
                  'num_classes: {}'.format(4), config)
  
  # Set batch size
  config = re.sub('batch_size: [0-9]+',
                  'batch_size: {}'.format(batch_size), config)
  
  # Set training steps
  config = re.sub('num_steps: [0-9]+',
                  'num_steps: {}'.format(num_steps), config)
  
  # Set fine-tune checkpoint type to detection
  config = re.sub('fine_tune_checkpoint_type: "classification"', 
             'fine_tune_checkpoint_type: "{}"'.format('detection'), config)
  
  f.write(config)

model_dir = 'training/'
pipeline_config_path = 'model_config.config'

## Train detector
!python /content/models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path={pipeline_config_path} \
    --model_dir={model_dir} \
    --alsologtostderr \
    --num_train_steps={num_steps} \
    --sample_1_of_n_eval_examples=1 \
    --num_eval_steps={num_eval_steps}

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir '/content/training/train'

## Export model inference graph
output_directory = 'inference_graph'
!python /content/models/research/object_detection/exporter_main_v2.py \
    --trained_checkpoint_dir {model_dir} \
    --output_directory {output_directory} \
    --pipeline_config_path {pipeline_config_path}

saved_model_path = '{output_directory}/saved_model/saved_model.pb'

## Test trained model on test images
## based on [Object Detection API Demo](https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/object_detection_tutorial.ipynb) and [Inference from saved model tf2 colab](https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/inference_from_saved_model_tf2_colab.ipynb).
"""

"""
Pour lancer ce programme d'inférence :

conda activate tf2
cd C:\tf2\workspace\training_demo
python tf2_inference_OD_model.py
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
matplotlib.use('Tkagg')  # affichage des images en mode GPU

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
    CSV_path = 'annotations/' + folder + '.csv'
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
  CSV_path = 'annotations/test.csv'
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

  """
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
  for i in range(min(max_boxes_to_draw, boxes.shape[0])):
      if scores is None or scores[i] > min_score_thresh:
          class_name = category_index[output_dict['detection_classes'][i]]['name']
          coordinates.append({
              "box": boxes,  # TODO construire liste JSON des prédictions avec le nom des images pour calculer les métriques : https://gist.github.com/tarlen5/008809c3decf19313de216b9208f3734
                             # + Groundtruths JSON aussi ==> Ligne 217 var. boxes
              "class_name": class_name,
              "score": scores[i]
          })
  print(coordinates)  # TODO JSON pour calculer les métriques ???
  """

  return output_dict, elapsed_time

def show_inference_perso(model, image_path, threshold, saveImg, espece):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image = Image.open(image_path)
    image_np = load_image_into_numpy_array(image) # numpy array with shape [height, width, 3]
    image_np_with_annotations = image_np.copy()

    # Load groundtruths : on ne garde que le nom de l'image courante à partir du chemin
    image_name = os.path.basename(image_path)
    groundtruths = load_groundtruths_on_Test(image_name)

    # Actual detection.
    output_dict, elapsed_time = run_inference_for_single_image(model, image_np_with_annotations, [])

    # Visualization of the results of a detection.
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_annotations,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        # on n'a pas plus de 3 ou 4 oiseaux en simultané
        #max_boxes_to_draw=4,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        line_thickness=4,
        use_normalized_coordinates=True,
        min_score_thresh=threshold,
        agnostic_mode=False)

    # Affichage des vérités terrain == annotations ou groundtruths
    fig, ax = plt.subplots()

    for gt in groundtruths:
      # Extrait les 4 coordonnées pour chaque box
      ymin, xmin, ymax, xmax, species = gt
      viz_utils.draw_bounding_box_on_image_array(
          image_np_with_annotations, 
          ymin, 
          xmin, 
          ymax, 
          xmax, 
          color='red', 
          thickness=2,
          use_normalized_coordinates=False)

      # Ajoute les vérités terrains des annotations, dans des rectangles en haut à droite des boxes annotées
      (left, right, top, bottom) = (xmin, xmax, ymin, ymax)

      # Positionne le rectangle dans le coin supérieur droit
      marginRight = 120
      rect = mpatches.Rectangle((right-marginRight, top), marginRight, 30, linewidth=2, edgecolor='r', facecolor='r')
      ax.add_patch(rect)
      ax.set_title('Appuyer sur "Q" pour défiler')

      # Ajout de l'espèce dans le rectangle
      rx, ry = rect.get_xy()
      cx = rx + rect.get_width()/2.0
      cy = ry + rect.get_height()/2.0
      ax.annotate(species, (cx, cy), color='black', weight='bold', 
                  fontsize=10, ha='center', va='center')

    # Affichage des labels en légende
    plt.legend(handles=handles)

    # Affichage ou sauvegarde des résultats
    if saveImg:
      # Créé un dossier pour sauvegarder l'espèce courante
      if not os.path.exists(espece):
        os.makedirs(espece)
      # Sauvegarde de l'image dans le dossier de l'espèce courante
      # Le dossier de la sauvegarde est situé au même endroit que ce programme Python !
      plt.imsave(espece + '\\' + image_name, image_np_with_annotations)
    else:
      # Affichage de l'image et des oiseaux reconnus avec leurs scores en plein écran
      wm = plt.get_current_fig_manager()
      wm.window.state('zoomed')
      plt.imshow(Image.fromarray(image_np_with_annotations))
      plt.show()

    return elapsed_time

# TODO remove ?
# Sur les N images aléatoires de test
"""
def inference_random_N_Images(test_images_dir, seuil, nb_img_inference):
  elapsed_time = []
  images_list = glob.glob(test_images_dir)
  # Inférence sur les N images
  for i in range(nb_img_inference):
    image_path = random.choice(images_list)
    elapsed_time = show_inference_perso(model, image_path, seuil)
  # Temps moyen (en secondes) par image pour la détéction et classification
  mean_elapsed = sum(elapsed_time) / float(len(elapsed_time))
  print('Elapsed time: ' + str(mean_elapsed) + ' second per image')
"""

# Sur les N images de CHAQUE ESPECE de test
def inference_random_N_Images_Especes(seuil, species_images, saveImg=None, espece=None):
  elapsed_time = []
  # Parcours des N images aléatoires pour l'espèce courante
  for i in species_images:
    image_path = os.getcwd() + '\\images\\test\\' + i
    elapsed_time = show_inference_perso(model, image_path, seuil, saveImg = saveImg, espece = espece)
  # Temps moyen (en secondes) par image pour la détéction et classification
  if elapsed_time != []:
    mean_elapsed = sum(elapsed_time) / float(len(elapsed_time))
    print('Elapsed time: ' + str(mean_elapsed) + ' second per image')

  #except ZeroDivisionError:
    #print('Le label : ', espece ,' est présent dans le fichier labelmap.txt est présent mais ne devrait pas y être !')

#########################################################################
######################### Partie  pour les tests #########################
#########################################################################
#########################################################################
# Chemins vers nos fichiers
#########################################################################
# TODO : valeurs par défaut + passage en param ...
current_dir = os.getcwd()
test_images_dir = current_dir + '\\images\\test\\'
train_record_path = current_dir + '\\annotations\\train.record'
test_record_path = current_dir + '\\annotations\\test.record'
labelmap_path = current_dir + '\\annotations\\labelmap.pbtxt'  # la labelmap d'entrainement sous forme d'objets

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
exported_model_folder = "exported-models\\my_mobilenet_model_2\\saved_model"
model = tf.saved_model.load(current_dir + '\\' + exported_model_folder)

#########################################################################
# Inférence sur N images random, parmi toutes les espèces
#########################################################################
"""
test_images_dir = current_dir + '\\images\\test\\*.jpg'
seuil = 0.65
N_images = 5  # N_images = len(os.listdir(test_images_dir))  si on veut le faire sur toutes les images de test
inference_random_N_Images(test_images_dir, seuil, N_images)
"""

#########################################################################
# Récupération de toutes les images d'une classe donnée, et inférence sur
# N images de chaque espèce !
# Remarque : Appuyer sur la touche 'Q' ou fermer la fenêtre d'affichage
# pour passer à l'image suivante
#########################################################################
# TODO : Code + propre avec les vars au dessus !!!
def N_random_per_species(N_images, saveImg=None):
  # Comptage des espèces et récupération des images par espèces
  bird_species_train, bird_species_test = count_images_by_species()
  # Inférence sur N images random de chaque espèce en TEST :
  for espece in labels:
    print('Espèce courante : ', espece)
    seuil = 0.60
    species_images = bird_species_test.get(espece)

    # Si on a au moins une image de cette espèce dans le jeu de données
    if list(species_images) != []:
      try:
        N_images_espece = random.sample(list(species_images), N_images)
      except ValueError:
        # Si on veut + d'images aléatoire que ce qu'on en a réellement, on prend le maximum possible
        N_images = len(list(species_images))
        N_images_espece = random.sample(list(species_images), N_images)

      # Sauvegarde les images avec les prédictions et annotations => Masque l'affichage graphique pour gagner du temps
      if saveImg:
        inference_random_N_Images_Especes(seuil, N_images_espece, saveImg, espece)
      else:
        # Ne sauvegarde pas les images avec les boxes prédites et annotées => affichage graphique des résultats
        inference_random_N_Images_Especes(seuil, N_images_espece, None, espece)

# TODO : passer en argument du programme les paramètres
#N_images = 10
N_images = len(os.listdir(test_images_dir))  # si on veut le faire sur toutes les images de test
#N_random_per_species(N_images, True) # si on sauvegarder les images avec les boxes prédites/annotées
N_random_per_species(N_images)  # sans sauvegarder les résultats de l'inférence

#########################################################################
# Moyenne de temps d'execution sur GPU pour 5 images :
#########################################################################
""" Elapsed time: 0.03820490837097168 second per image """

# PRIO 1 !
"""
OK // Tableau métriques AP, nb_images TRAIN, nb_images TEST ; avoir la précision/recall  
OK // SAVE MESCHA, ecurou, mesble, vereur predictions sur toutes les images dispos
OK // courbes LOSS trainV2

Regarder si notre modèle peut fonctionner sur RPi en s'inspirant de ces tutos officiels :
https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/eager_few_shot_od_training_tflite.ipynb
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tf2.md
"""

# TODO : Evaluer selon chaque espèce, selon le sexe, ...
# ==> avec plusieurs matrices de confusions ?
# ==> Pour évaluer selon espèces => données triées avec train_test_split()
#     même méthode si on veut évaluer les données selon le sexe ? => !!! trier les sexes "unknown" comme pour les espèces dans xml_to_csv.py

# ==> Données de validation ? pour voir les val_loss et acc_loss sur Tensorboard et détecter l'overfitting ?
# train_test_split() mélange aléatoirement toutes les données avant de les séparer en 2 groupes : TRAIN et TEST
# TODO : Comparer les métriques obtenues avec train_test_split() VS K-Fold qui permet d'avoir différents "jours" en TRAIN et en TEST

# TODO Vérif :
# 2021-01-05-16-15-17.jpg ça aussi c'est une bleu
# ça 2021-01-06-12-12-47.jpg ça ressemble effectivement plus à de la bleu
# sur 2021-01-06-11-12-29.jpg

# Compute IoU manually from BBox coordinates : # https://towardsdatascience.com/intersection-over-union-iou-calculation-for-evaluating-an-image-segmentation-model-8b22e2e84686  
                                               # https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
                                               # OU  https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
# ==> Détails métrique mAP  :  cf OneNote
# TODO : Exporter la commande en sortie dans un fichier texte puis PARSER et créer un tableau Excel (ou autre) à partir des métriques en console
# soit je copie colle la console et je parse le fichier
# ou je le fais à la main si on a peu de métriques : Recall, Precision, IoU -> TODO CoCo à éclaircir
# comment les scores de confiance sont calculés % à l'IoU ?

"""
Accuracy, IoU, globale, par espèces, par attribut: sex, behaviour..., avec différent training design: kfold sur les jours/site, random etc...)
==> On a l'IoU globale et le mAP (cf image paint)
=> Plusieurs TRAIN ==> scripts CALMIP ?
"""