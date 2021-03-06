#!/bin/bash

#SBATCH -N 2
#SBATCH -n 72
#SBATCH --ntasks-per-node=36
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --time=10:00:00
#SBATCH --mail-user=floflu19@gmail.com
#SBATCH --mail-type=END

# Charge les modules pr�d�finis
module purge
module load conda/4.9.2
module load cuda/11.2

# Active notre environnement virtuel CONDA
conda activate tf2

# Lien vers le fichier python � executer
EXE=/tmpdir/DONNEEST21001/tf2/workspace/training_demo/exporter_main_v2.py

# Variables pour les param�tres du programme python � modifier en fonction du mod�le entrain� !!
# Nom du dossier pour l'exportation � modifier (SSD_64 ici) !!
INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=/tmpdir/DONNEEST21001/tf2/workspace/training_demo/models/SSD_64/pipeline.config
TRAINED_CKPT_PREFIX=/tmpdir/DONNEEST21001/tf2/workspace/training_demo/models/SSD_64

# Dossier pour l'exportation du mod�le
# mkdir -p cr�er se dossier s'il n'existe pas
EXPORT_DIR=/tmpdir/DONNEEST21001/tf2/workspace/training_demo/exported-models/SSD_64
mkdir -p $EXPORT_DIR

#######################################
# Exporte le mod�le pour Tensorflow 2 #
#######################################
# Exporte les derniers poids du mod�le (version TF2)
python $EXE \
--input_type=${INPUT_TYPE} \
--pipeline_config_path=${PIPELINE_CONFIG_PATH} \
--trained_checkpoint_dir=${TRAINED_CKPT_PREFIX} \
--output_directory=${EXPORT_DIR}

##########################################
# Exporte le mod�le pour Tensorflow Lite #
##########################################
# Lien vers le fichier python � executer
EXE=/tmpdir/DONNEEST21001/tf2/workspace/training_demo/export_tflite_graph_tf2.py

# Variables pour les param�tres du programme python � modifier en fonction du mod�le entrain� !!
# Nom du dossier pour l'exportation � modifier (SSD_64_lite ici) !!
# mkdir -p cr�er se dossier s'il n'existe pas
EXPORT_DIR=/tmpdir/DONNEEST21001/tf2/workspace/training_demo/exported-models/SSD_64_lite
mkdir -p $EXPORT_DIR

MODEL_WEIGHTS=/tmpdir/DONNEEST21001/tf2/workspace/training_demo/exported-models/SSD_64_lite/saved_model

# Exporte les derniers poids du mod�le (version TF Lite) avec les m�mes param�tres qu'au d�but
python $EXE \
--pipeline_config_path=${PIPELINE_CONFIG_PATH} \
--trained_checkpoint_dir=${TRAINED_CKPT_PREFIX} \
--output_directory=${EXPORT_DIR}

# Cr�ation du fichier .tflite avec les param�tres ci-dessus
EXE=/tmpdir/DONNEEST21001/tf2/workspace/training_demo/convert-to-tflite.py

python $EXE \
--model=${MODEL_WEIGHTS}  \
--output=${EXPORT_DIR}

# END OF SBATCH SCRIPT ----------------
