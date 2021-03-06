# -*- coding: utf-8 -*-
'''
Purpose: Split the dataset into train and test sets
'''
import os
from sklearn.model_selection import train_test_split
import argparse
from tqdm import tqdm
from xml.etree import ElementTree
from shutil import copyfile

# TODO ajouter UN PARAM DE PROG pour => MESCHA, ... etc si on souhaite les exclure de l'entrainement & random_state value
liste_especes = []

def parse_UnwantedLabels(root):
    # On ignore ces labels pour notre jeu de données
    unwanted_labels = ['noBird', 'human', 'unknown']
    # Recherche de la présence d'un oiseau ou d'un label non souhaité
    names = root.findall('.//attribute/name')
    values = root.findall('.//attribute/value')
    # Récupère les indices de toutes les espèces dans le fichier .xml
    indices = [i for i, x in enumerate(names) if x.text == "species"]
    # Recherche d'un label non souhaité
    # True si on a une espèce indésirée, False sinon
    for i in indices :
        # Collecte toutes les espèces uniques
        if values[i].text not in liste_especes and values[i].text not in unwanted_labels:
            liste_especes.append(values[i].text)

        # Si c'est une espèce non souhaitée, on ne la garde pas
        if values[i].text in unwanted_labels:
            return True
    return False

def parse_tasks(chemin, annots_valides, TASKS_TEST):
    train_files = []
    test_files = []
    # parcours de tous les fichiers d'annotations
    for filename in os.listdir(chemin):
        ann_path = chemin + filename
        tree = ElementTree.parse(ann_path)
        root = tree.getroot()

        # Récupère le nom de la tâche
        full_filename = root.find('.//filename').text
        # Récupère le nom de la task
        task_name = full_filename.split('/')[1]  # exemple : <filename>bird/task_05-01-2021/2021-01-05-15-58-48.jpg</filename>

        with open(os.path.join(chemin, filename), 'r') as f: # open in readonly mode
            # Si ce n'est pas un fichier XML mal annoté ou qui contient des classes non souhaitées
            if filename in annots_valides:
                # Si le nom de la task contient une sous-partie en paramètre, elle est pour le TEST
                is_test = False
                for s in TASKS_TEST:
                    if task_name.find(s) != -1:
                        is_test = True
                # Ajoute le fichier au TEST
                if is_test:
                    test_files.append(filename)
                else:
                    # Sinon on a une task de TRAIN
                    train_files.append(filename)
                    
    return train_files, test_files

# Chemin depuis le dossier courant du fichier vers les annotations .xml
def detectBadAnnotations(chemin):
    invalid_files = []
    # parcours de tous les fichiers d'annotations
    for filename in os.listdir(chemin):
        ann_path = chemin + filename
        tree = ElementTree.parse(ann_path)
        root = tree.getroot()

        with open(os.path.join(chemin, filename), 'r') as f: # open in readonly mode
            # Si l'image n'est pas annotée ou qu'il n'y a pas d'oiseau, ou s'il y a un humain, 
            # alors cette image est incorrecte pour notre dataset
            if not root.findall('.//bndbox') or parse_UnwantedLabels(root):
                # on garde juste les noms sans extension
                invalid_files.append(filename)
    return invalid_files

# Lecture des arguments passés au programme en ligne de commande
argparser = argparse.ArgumentParser(description='Split dataset into train and test set')
argparser.add_argument('-a', '--annotations',
                       help='path to annotations\' directory')
argparser.add_argument('-i', '--images',
                       help='path to images\' directory')
argparser.add_argument('-o', '--outputdir',
                       help='where do you want your train and test directories?')
argparser.add_argument('-s', '--testsize',
                       help='test set size % (0 to 1)',
			           default=0.2)
argparser.add_argument('-t', '--tasks',
                       nargs="+",
                       help='tasks that are in the TEST folder, the remainings are for TRAIN',
			           default=[])

args = argparser.parse_args()

# parse arguments
ANNOTATIONS = args.annotations
IMAGES = args.images
OUTPUT_DIR = args.outputdir
TEST_SET_SIZE = float(args.testsize)
TASKS_TEST = args.tasks

# Vérifie qu'il ne manque pas d'annotations dans nos .xml pour TRAIN et TEST :
invalid_files = detectBadAnnotations(ANNOTATIONS)
print('\nOn ignore les images avec des annotations indésirables : ', len(invalid_files))
print('Liste des espèces présentes : ', liste_especes)   # TODO automatiser labelmap ?

# create train and test directories
if not os.path.isdir(os.path.join(OUTPUT_DIR, "train")):
    os.makedirs(os.path.join(OUTPUT_DIR, "train"))
    print("Created {} directory".format(os.path.join(OUTPUT_DIR, "train")))

if not os.path.isdir(os.path.join(OUTPUT_DIR, "test")):
    os.makedirs(os.path.join(OUTPUT_DIR, "test"))
    print("Created {} directory".format(os.path.join(OUTPUT_DIR, "test")))

# On ne copie que les annotations souhaitées, sinon on les ignore
liste_annotations = os.listdir(ANNOTATIONS)
annots_valides = [item for item in liste_annotations if item not in invalid_files]

# split the data into test and train
X = y = annots_valides
# Attention car on n'a pas un nombre d'images NON homogène pour toutes les classes ==> Diversité (imbalanced/unbalanced classes) dans le dataset
# Si on n'a pas précisé de liste de tasks CVAT pour le TEST :
if not TASKS_TEST :
    # Split random et reproductible
    # seed = 1 pour reproduire le jeu de données aléatoire de TrainV2, V4, V8
    # seed = 42 pour reproduire le jeu de données aléatoire de TrainV3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SET_SIZE, random_state=1)
else:
    # Parcourt les fichiers XML pour trier les tasks en paramètre en TEST, et les autres en TRAIN
    X_train, X_test = parse_tasks(ANNOTATIONS, annots_valides, TASKS_TEST)

print("\nTraining set size: ", len(X_train), "\nTest set size: ", len(X_test))

# copy the files according to the split
bad_files = 0
pbar = tqdm(total=len(X_train), position=1, desc="Copying train set..")
for f in X_train:
    pbar.update(1)

    try:
        # copy annotations
        copyfile(os.path.join(ANNOTATIONS, f), os.path.join(OUTPUT_DIR, 'train', f))
        # copy image
        img_file = f.replace(".xml", ".jpg")
        copyfile(os.path.join(IMAGES, img_file), os.path.join(OUTPUT_DIR, 'train', img_file))
    except:
        bad_files += 1

pbar = tqdm(total=len(X_test), position=3, desc="Copying test set..")
for f in X_test:
    pbar.update(1)

    try:
        # copy annotations
        copyfile(os.path.join(ANNOTATIONS, f), os.path.join(OUTPUT_DIR, 'test', f))
        # copy image
        img_file = f.replace(".xml", ".jpg")
        copyfile(os.path.join(IMAGES, img_file), os.path.join(OUTPUT_DIR, 'test', img_file))
    except:
        bad_files += 1

print("\n\nBad files count: ", bad_files) 