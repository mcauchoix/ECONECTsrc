import os
import pandas as pd

def main():
    for folder in ['train','test']:
        CSV_path = '../../workspace/training_demo/annotations/' + folder + '.csv'
        # Comptage des espèces :
        df = pd.read_csv(CSV_path)
        class_count = df["class"].value_counts()
        print("Pour le " + folder + " :\n", class_count)
main()
