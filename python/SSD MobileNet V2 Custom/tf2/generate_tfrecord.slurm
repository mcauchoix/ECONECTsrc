#! /bin/bash

#SBATCH -N 2
#SBATCH -n 72
#SBATCH --ntasks-per-node=36
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --time=1:00:00

# Charge les modules pr�d�finis
module purge
module load conda/4.9.2 
module load cuda/11.2
conda activate tf2

# Lien vers le fichier python � executer
EXE=/tmpdir/DONNEEST21001/tf2/scripts/preprocessing/generate_tfrecord.py

# G�n�ration des fichiers CSV et .tfrecord pour le TRAIN
echo 'Generating CSV and .tfrecord files ...'
python $EXE \
-x /tmpdir/DONNEEST21001/tf2/workspace/training_demo/images/train \
-l /tmpdir/DONNEEST21001/tf2/workspace/training_demo/annotations/labelmap_14especes.pbtxt \
-o /tmpdir/DONNEEST21001/tf2/workspace/training_demo/annotations/train.record \
-c /tmpdir/DONNEEST21001/tf2/workspace/training_demo/annotations/train.csv

# G�n�ration des fichiers CSV et .tfrecord pour le TEST
python $EXE \
-x /tmpdir/DONNEEST21001/tf2/workspace/training_demo/images/test \
-l /tmpdir/DONNEEST21001/tf2/workspace/training_demo/annotations/labelmap_14especes.pbtxt \
-o /tmpdir/DONNEEST21001/tf2/workspace/training_demo/annotations/test.record \
-c /tmpdir/DONNEEST21001/tf2/workspace/training_demo/annotations/test.csv


# END OF SBATCH SCRIPT ----------------

