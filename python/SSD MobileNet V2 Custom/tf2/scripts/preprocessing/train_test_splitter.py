'''
Purpose: Split the dataset into train and test sets
'''
import os
from sklearn.model_selection import train_test_split
import argparse
from tqdm import tqdm
from xml.etree import ElementTree
from shutil import copyfile

def ignore_badLabels(root):
    # On ignore ces labels pour notre jeu de données
    unwanted_labels = ['noBird', 'human', 'unknown', 'MESCHA']  # ajouter MESCHA si on souhaite les exclure de l'entrainement
    # Recherche de la présence d'un oiseau ou d'un label non souhaité
    names = root.findall('.//attribute/name')
    values = root.findall('.//attribute/value')
    # Récupère les indices de toutes les espèces dans le fichier .xml
    indices = [i for i, x in enumerate(names) if x.text == "species"]
    # Recherche d'un label non souhaité
    # True si on a une espèce indésirée, False sinon
    for i in indices :
        if values[i].text in unwanted_labels:
            return True
    return False

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
            if not root.findall('.//bndbox') or ignore_badLabels(root):
                # on garde juste les noms sans extension
                invalid_files.append(filename[:-4])
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
			default=0.1)

args = argparser.parse_args()

# parse arguments
ANNOTATIONS = args.annotations
IMAGES = args.images
OUTPUT_DIR = args.outputdir
TEST_SET_SIZE = float(args.testsize)

# Vérifie qu'il ne manque pas d'annotations dans nos .xml pour TRAIN et TEST :
invalid_files = detectBadAnnotations(ANNOTATIONS)
print('On ignore les images avec des annotations indésirables : ', len(invalid_files))

# create train and test directories
if not os.path.isdir(os.path.join(OUTPUT_DIR, "train")):
    os.makedirs(os.path.join(OUTPUT_DIR, "train"))
    print("\nCreated {} directory\n".format(os.path.join(OUTPUT_DIR, "train")))

if not os.path.isdir(os.path.join(OUTPUT_DIR, "test")):
    os.makedirs(os.path.join(OUTPUT_DIR, "test"))
    print("\nCreated {} directory\n".format(os.path.join(OUTPUT_DIR, "test")))

# get annotations only ending with '.xml'
annots = []
for filename in os.listdir(ANNOTATIONS):
    # On ne copie que les images avec leurs annotations qui correspondent à nos classes souhaitées, sinon on les ignore
	if filename.endswith('.xml') and filename[:-4] not in invalid_files :
		annots.append(filename)

# split the data into test and train
X = y = annots
# Attention car on n'a pas un nombre d'images homogène pour toutes les classes ==> Diversité (imbalanced/unbalanced classes) dans le dataset
# seed = 1 pour reproduire le jeu de données aléatoire de TrainV2
# seed = 42 pour reproduire le jeu de données aléatoire de TrainV3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SET_SIZE, random_state=1)  
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