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

# Variable d'environnement pour Matplotlib
export MPLBACKEND=Qt5Agg

# Lien vers le fichier python à executer
EXE=/tmpdir/DONNEEST21001/tf2/workspace/training_demo/tf2_inference_OD_model.py

# Inférence avec TF2
python $EXE

# END OF SBATCH SCRIPT ----------------

