#!/bin/bash

#SBATCH -N 2
#SBATCH -n 72
#SBATCH --ntasks-per-node=36
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --time=10:00:00
#SBATCH --mail-user=floflu19@gmail.com
#SBATCH --mail-type=END

# Charge les modules prédéfinis
module purge
module load conda/4.9.2
module load cuda/11.2

# Active notre environnement virtuel CONDA
conda activate tf2

# Lien vers le fichier python à executer
EXE=/tmpdir/DONNEEST21001/tf2/workspace/training_demo/exporter_main_v2.py

# Variables pour les paramètres du programme python à modifier en fonction du modèle entrainé !!
# Nom du dossier pour l'exportation à modifier (SSD_8 ici) !!
INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=/tmpdir/DONNEEST21001/tf2/workspace/training_demo/models/SSD_8/pipeline.config
TRAINED_CKPT_PREFIX=/tmpdir/DONNEEST21001/tf2/workspace/training_demo/models/SSD_8

# Dossier pour l'exportation du modèle
# mkdir -p créer se dossier s'il n'existe pas
EXPORT_DIR=/tmpdir/DONNEEST21001/tf2/workspace/training_demo/exported-models/SSD_8
mkdir -p $EXPORT_DIR

#######################################
# Exporte le modèle pour Tensorflow 2 #
#######################################
# Exporte les derniers poids du modèle (version TF2)
python $EXE \
--input_type=${INPUT_TYPE} \
--pipeline_config_path=${PIPELINE_CONFIG_PATH} \
--trained_checkpoint_dir=${TRAINED_CKPT_PREFIX} \
--output_directory=${EXPORT_DIR}
echo 'Fin 1'

##########################################
# Exporte le modèle pour Tensorflow Lite #
##########################################
# Lien vers le fichier python à executer
EXE=/tmpdir/DONNEEST21001/tf2/workspace/training_demo/export_tflite_graph_tf2.py

# Variables pour les paramètres du programme python à modifier en fonction du modèle entrainé !!
# Nom du dossier pour l'exportation à modifier (SSD_8_lite ici) !!
# mkdir -p créer se dossier s'il n'existe pas
EXPORT_DIR=/tmpdir/DONNEEST21001/tf2/workspace/training_demo/exported-models/SSD_8_lite
mkdir -p $EXPORT_DIR

MODEL_WEIGHTS=/tmpdir/DONNEEST21001/tf2/workspace/training_demo/exported-models/SSD_8_lite/saved_model

# Exporte les derniers poids du modèle (version TF Lite) avec les mêmes paramètres qu'au début
python $EXE \
--pipeline_config_path=${PIPELINE_CONFIG_PATH} \
--trained_checkpoint_dir=${TRAINED_CKPT_PREFIX} \
--output_directory=${EXPORT_DIR}
echo 'Fin 2'

# Création du fichier .tflite avec les paramètres ci-dessus
EXE=/tmpdir/DONNEEST21001/tf2/workspace/training_demo/convert-to-tflite.py

python $EXE \
--model=${MODEL_WEIGHTS}  \
--output=${EXPORT_DIR}
echo 'Fin 3'

# END OF SBATCH SCRIPT ----------------
