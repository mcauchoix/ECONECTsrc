#! /bin/bash

#SBATCH -N 1
#SBATCH -n 9
#SBATCH --ntasks-per-core=1
#SBATCH --mem 50G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

# Charge les modules prédéfinis
module purge
module load conda/4.9.2 
module load cuda/11.2
conda activate tf2

# Lien vers le fichier python à executer
EXE=/tmpdir/DONNEEST21001/tf2/scripts/preprocessing/generate_tfrecord.py

# Génération des fichiers CSV et .tfrecord pour le TRAIN
echo 'Generating CSV and .tfrecord files ...'
python $EXE \
-x /tmpdir/DONNEEST21001/tf2/workspace/training_demo/images/train \
-l /tmpdir/DONNEEST21001/tf2/workspace/training_demo/annotations/labelmap_14especes.pbtxt \
-o /tmpdir/DONNEEST21001/tf2/workspace/training_demo/annotations/train.record \
-c /tmpdir/DONNEEST21001/tf2/workspace/training_demo/annotations/train.csv

# Génération des fichiers CSV et .tfrecord pour le TEST
python $EXE \
-x /tmpdir/DONNEEST21001/tf2/workspace/training_demo/images/test \
-l /tmpdir/DONNEEST21001/tf2/workspace/training_demo/annotations/labelmap_14especes.pbtxt \
-o /tmpdir/DONNEEST21001/tf2/workspace/training_demo/annotations/test.record \
-c /tmpdir/DONNEEST21001/tf2/workspace/training_demo/annotations/test.csv


# END OF SBATCH SCRIPT ----------------

