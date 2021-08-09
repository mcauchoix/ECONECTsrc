import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

# Converti les annotations .xml en un seul fichier CSV : un CSV pour le train et un autre pour le test
def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            bbx = member.find('bndbox')
            xmin = int(float(bbx.find('xmin').text))
            ymin = int(float(bbx.find('ymin').text))
            xmax = int(float(bbx.find('xmax').text))
            ymax = int(float(bbx.find('ymax').text))
            species = member.find('attributes')
            label = species[3][1].text

            filename_path = root.find('filename').text
            filename = filename_path.split('/')[-1]
            value = (filename,
                     int(float(root.find('size')[0].text)),
                     int(float(root.find('size')[1].text)),
                     label,
                     xmin,
                     ymin,
                     xmax,
                     ymax
                     )
            xml_list.append(value)

    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    for folder in ['train','test']:
        # Création des fichiers CSV pour train et test
        #image_path = os.path.join(os.getcwd(), ('images/' + folder))
        #xml_df = xml_to_csv(image_path)
        CSV_path = '../../workspace/training_demo/annotations/' + folder + '.csv'
        #xml_df.to_csv(CSV_path, index=None)
        #print('Successfully converted xml to csv.')

        # Comptage des espèces :
        df = pd.read_csv(CSV_path)
        class_count = df["class"].value_counts()
        print("Pour le " + folder + " :\n", class_count)
main()
