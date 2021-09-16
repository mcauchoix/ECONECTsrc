#!/bin/bash

#SBATCH -N 2
#SBATCH -n 72
#SBATCH --ntasks-per-node=36
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --mem 50G
#SBATCH --time=01:00:00

# Charge les modules prédéfinis
module purge
module load conda/4.9.2 cuda/11.0

# Active notre environnement virtuel CONDA
source /usr/local/miniconda/4.9.2/bin/activate tf2

# Lien vers le fichier python à executer
EXE=/tmpdir/DONNEEST21001/tf2/workspace/training_demo/model_main_tf2.py

# Génération des fichiers CSV et .tfrecord pour le TRAIN
echo 'Training phase ...'
python $EXE \
  --model_dir=/tmpdir/DONNEEST21001/tf2/workspace/training_demo/models/SSD_Mbnet_v2_CVAT_07_09 \
  --pipeline_config_path=/tmpdir/DONNEEST21001/tf2/workspace/training_demo/models/SSD_Mbnet_v2_CVAT_07_09/pipeline.config

# END OF SBATCH SCRIPT ----------------

