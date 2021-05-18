"""
D:
cd D:\M2_IARF\_Stage\MobileNetSSD
Exemples pour lancer le script : 
    python image_similarity.py Biodicam_IMG
    python image_similarity.py 2021-02-28-FRANCON
"""
# https://idealo.github.io/imagededup/methods/cnn/
# https://idealo.github.io/imagededup/user_guide/finding_duplicates/
from imagededup.methods import CNN
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore") #ignore les warnings de tensorflow dus aux différentes versions
import argparse
import os
from pathlib import Path
import shutil
import time

# Créé des sous-dossiers pour chaque jour différent avec les images correspondantes
def createFolders_FindSimilarities(image_dir, seuil):
    # Parcours des images
    for img in os.listdir(image_dir):
        current_folder = image_dir + img
        name, ext = os.path.splitext(img)
        # Si c'est une image, on créé le dossier correspondant à son jour et on la déplace dedans
        if ext == '.jpg':
            current_date = img.split('-')[0]
            # Créé le dossier associé à la date courante (s'il n'existe pas déjà)
            newImage_folder = image_dir + 'task_' + current_date
            Path(newImage_folder).mkdir(parents=True, exist_ok=True)
            # Et déplace toutes les images du jour courant dans le sous-dossier créé
            shutil.move(current_folder, newImage_folder)
    # Ensuite pour chaque sous-dossier, on cherche les images dupliquées
    for root, dirs, files in os.walk(image_dir):
        for dir in dirs:
            if dir != "trash":
                current_path = image_dir + dir + "\\"
                # Recherche des images dupliquées dans chaque dossier :
                trierDuplicates(current_path, seuil)

def trierDuplicates(image_dir, seuil):
    cnn = CNN()
    encodings = cnn.encode_images(image_dir=image_dir)
    # Trouve les images trop ressemblantes à 95% ou + pour ne pas manquer de petits oiseaux !
    trash_folder_path = image_dir + "\\trash\\"
    if not os.path.exists(trash_folder_path):
        # Et on fait le scan uniquement si on ne l'a pas déjà fait
        duplicates = cnn.find_duplicates_to_remove(image_dir=image_dir, encoding_map=encodings,
                                                   min_similarity_threshold=seuil)
        nbDupliquees = len(duplicates)
        # Créé le sous-dossier pour stocker les images dupliquées
        os.makedirs(trash_folder_path)
        # Déplace les images dupliquées
        for imgName in duplicates:
            shutil.move(image_dir + imgName, trash_folder_path + imgName)

# Move files into subdirectories
def move_files(abs_dirname, N):
    root = Path(abs_dirname).parent.absolute()
    current_dateDir = abs_dirname.split('\\')[-1] + "_"
    files = [os.path.join(abs_dirname, f) for f in os.listdir(abs_dirname)]
    i = 0
    curr_subdir = None
    # Parcours des images de la task courante (images triées par jour)
    for f in files:
        # create new subdir if necessary
        if i % N == 0:
            # On ajoute un numéro (001, 002, ...) au dossier de la task qui contient plus de 100 images
            # exemple : task_20210409_001
            # on place ces dossiers dans la racine (1 niveau au dessus le dossier courant)
            subdir_name = os.path.join(root, current_dateDir + '{0:03d}'.format(i // N + 1))
            os.mkdir(subdir_name)
            curr_subdir = subdir_name
        # move file to current dir
        f_base = os.path.basename(f)
        shutil.move(f, os.path.join(subdir_name, f_base))
        i += 1
    # Une fois que l'on a crée les sous-dossiers pour la task courante, on supprime le dossier de cette task (exemple : task_20210409) car il est vide
    shutil.rmtree(abs_dirname)

def parse_args():
    """ Paramètre à passer lorsqu'on lance le programme : 
            nom du dossier dans lequel on fait la détéction d'images dupliquées,
            ce dossier doit se situer au même niveau hiérarchique que ce fichier python
        exemple : python image_similarity.py Biodicam_IMG
    """
    # TODO lien absolu pour éviter de placer le fichier python juste avant le bon dossier ? + ligne 134 du main
    parser = argparse.ArgumentParser(description='Split files into multiple subfolders.')
    parser.add_argument('src_dir', help='repertoire source')
    return parser.parse_args()

def countAndDeleteBadImages(image_dir):
    # Parcours des sous-dossiers avec les dates des images
    goodImages = 0; badImages = 0; totalGood = 0; totalBad = 0; i = 0; cheminSupprImages = ""
    for folder in os.listdir(image_dir):
        goodImages = len(os.listdir(image_dir + folder)) - 1
        totalGood += goodImages
        print("Folder : ", folder, " -> ", goodImages, " images gardées")
        currentFolder = image_dir + folder

        # Parcours des sous-dossiers et recherche du dossier 'trash'
        for root, dirs, files in os.walk(image_dir + folder):  # TODO comme ça et suppr le for au dessus
            for dir in dirs:
                if dir == 'trash':
                    cheminSupprImages = currentFolder + "\\" + dir + "\\"
                    badImages = len(os.listdir(cheminSupprImages))
                    totalBad += badImages
                    print("Images dupliquées (supprimées) : ", badImages)

        # Une fois que l'on a supprimé toutes les mauvaises images, on supprime le dossier 'trash' avec les images trop similaires
        if os.path.exists(cheminSupprImages):
            shutil.rmtree(cheminSupprImages)
        # Ensuite, on déplace les images sauvegardées pour créer des paquets de 100 et les uploader sur Gitlab !
        # On créé des paquets de 100 images, qui sont triées par date
        if len(os.listdir(currentFolder)) > 100:
            # Création des sous-dossiers pour les paquets de 100 images
            N = 100  # nombre d'images par sous-dossier
            move_files(os.path.abspath(currentFolder), N)
            i += 1   
    print("Nombre total d'images gardées : ", totalGood, " sur ", totalBad+totalGood, " soit ", (totalGood/(totalBad+totalGood))*100, "%")

def main():
    args = parse_args()
    src_dir = args.src_dir
    # Gère le cas où le dossier n'existe pas / mauvais chemin
    if not os.path.exists(src_dir):
        raise Exception('Directory does not exist ({0}).'.format(src_dir))

    image_dir = os.getcwd() + "\\" + src_dir + "\\"
    # seuil de similarité pour restreindre beaucoup, sinon 0.90 ou 0.95 pour ne pas manquer d'oiseau
    seuil = 0.90  # 0.2 ou 0.4 à 0.5 si on veut restreindre davantage sur les 23000 images de l'UPS
    createFolders_FindSimilarities(image_dir, seuil)
    countAndDeleteBadImages(image_dir)

if __name__ == '__main__':
    main()