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

## Test trained model on test images
## based on [Object Detection API Demo](https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/object_detection_tutorial.ipynb) and [Inference from saved model tf2 colab](https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/inference_from_saved_model_tf2_colab.ipynb).

"""
Pour lancer ce programme d'inférence :
cd /tmpdir/DONNEEST21001/tf2
sbatch tf2_inference.slurm
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
  for i in range(min(max_boxes_to_draw, boxes.shape[0])):
      if scores is None or scores[i] > min_score_thresh:
          class_name = category_index[output_dict['detection_classes'][i]]['name']
          coordinates.append({
              "box": boxes,  # TODO construire liste JSON des prédictions avec le nom des images pour calculer les métriques : https://gist.github.com/tarlen5/008809c3decf19313de216b9208f3734
                             # + Groundtruths JSON aussi ==> Ligne 217 var. boxes
              "class_name": class_name,
              "score": scores[i]
          })
  
  #print(coordinates)  # TODO JSON pour calculer les métriques ???

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

    #TODO
    img_annotee = Image.fromarray(np.uint8(image_np_with_annotations)).convert('RGB')
    draw = ImageDraw.Draw(img_annotee)
    for gt in groundtruths:
      ymin, xmin, ymax, xmax, species = gt
      (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
      marginRight = 145
      marginBottom = 30
      """
      draw.rectangle(
          [(right, bottom), (right, bottom)],
          outline='red', fill='red')
      """
      draw.text(
          (right-marginRight, bottom-marginBottom),
          species,
          fill='white',
          font = ImageFont.truetype(current_dir + "arial.ttf", 28))

    #img_annotee.show()
    #input('W8')
    # Copie les vérités terrain sur l'image avec les prédictions
    np.copyto(image_np_with_annotations, np.array(img_annotee)) 

    # Affichage des labels en légende
    plt.legend(handles=handles)

    # Affichage ou sauvegarde des résultats
    if saveImg:
      # Créé un dossier pour sauvegarder l'espèce courante
      if not os.path.exists(current_dir + espece):
        os.makedirs(current_dir + espece)
      # Sauvegarde de l'image dans le dossier de l'espèce courante
      # Le dossier de la sauvegarde est situé au même endroit que ce programme Python !
      plt.subplots_adjust(right=0.7)
      plt.imsave(current_dir + espece + '/' + image_name, image_np_with_annotations)
    else:
      """
      # Ajoute les vérités terrains des annotations, dans des rectangles en haut à droite des boxes annotées
      fig, ax = plt.subplots()
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
      """

      # Affichage de l'image et des oiseaux reconnus avec leurs scores en plein écran
      #wm = plt.get_current_fig_manager()
      #wm.window.state('zoomed')
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
    image_path = current_dir + '/images/test/' + i
    elapsed_time = show_inference_perso(model, image_path, seuil, saveImg = saveImg, espece = espece)
  # Temps moyen (en secondes) par image pour la détéction et classification
  if elapsed_time != []:
    mean_elapsed = sum(elapsed_time) / float(len(elapsed_time))
    print('Elapsed time: ' + str(mean_elapsed) + ' second per image')

#########################################################################
######################### Partie  pour les tests ########################
#########################################################################
#########################################################################
# Chemins vers nos fichiers
#########################################################################
# TODO : valeurs par défaut + passage en param ...
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
exported_model_name = 'SSD_8'
exported_model_folder = "/exported-models/" + exported_model_name + "/saved_model"
model = tf.saved_model.load(current_dir + exported_model_folder)

#########################################################################
# Inférence sur N images random, parmi toutes les espèces
#########################################################################
"""
test_images_dir = current_dir + '/images/test/*.jpg'
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
        inference_random_N_Images_Especes(seuil, N_images_espece)
        # TODO prendre une liste d'images en entrée : ex => tous les MOIDOM_A à 1 dans le dataframe pandas
        # TODO : possibilité de choisir une liste d'img ['2021-06-12-06-09-15.jpg', '2021-06-12-06-27-31.jpg', '2021-06-12-06-27-31.jpg'] au lieu de N_images_espece
        # 3 MESCHA : ['2021-06-12-06-09-15.jpg', '2021-06-12-06-27-31.jpg', '2021-06-12-06-27-31.jpg']
        # 4 MESCHA : ['2021-06-12-06-10-03.jpg']
        # Les 3 MOIDOM : ['20210227-115654_(8.0).jpg', '20210227-115733_(6.0).jpg', '20210301-090723_(9.0).jpg']
        # 3 MESCHA en inférence pour comparer les SSD : ['2021-06-12-06-10-07.jpg']
        #inference_random_N_Images_Especes(seuil, ['2021-06-12-06-10-03.jpg', '2021-06-12-06-10-07.jpg'])

# TODO : passer en argument du programme les paramètres
#N_images = 10
N_images = len(os.listdir(test_images_dir))  # si on veut le faire sur toutes les images de test

# Pour CALMIP, le comportement par défaut est de sauvegarder les détections sur les images
# Car il n'y a pas possibilité d'afficher des résultats graphiques
N_random_per_species(N_images, True) # si on sauvegarde les images avec les boxes prédites/annotées