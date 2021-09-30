#! /bin/bash

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
conda activate tf2

# Variable d'environnement graphique pour Matplotlib
# export MPLBACKEND=Qt5Agg

# Inférence avec TF2
# TODO : Fixer le problème de sauvegarde des détections
EXE=/tmpdir/DONNEEST21001/tf2/workspace/training_demo/tf2_inference_OD_model.py
echo 'Saving detections for each species ...'
python $EXE

# TODO paramètres de prog avec le dossier du modèle, ...
# Evaluation avec TF2
EXE=/tmpdir/DONNEEST21001/tf2/workspace/training_demo/tf2_matrice_confusion.py
echo 'Generating CSV to calculate metrics (precision and recall) ...'
python $EXE

# END OF SBATCH SCRIPT ----------------

