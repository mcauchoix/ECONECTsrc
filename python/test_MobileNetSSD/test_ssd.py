import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
import os
import random
import numpy as np
import time
import PIL.Image as Image
import matplotlib.pyplot as plt

'''
conda activate ro
D:
cd D:\M2_IARF\_Stage\Romain
python test_ssd.py
'''

model = 'https://tfhub.dev/google/aiy/vision/classifier/birds_V1/1'

IMAGE_SHAPE = (224, 224)
classifier = tf.keras.Sequential([
    hub.KerasLayer(model, input_shape=IMAGE_SHAPE+(3,))
])

img_dir = os.getcwd() + "\\tosend\\"
random_img = random.choice([x for x in os.listdir(img_dir)])
bird = Image.open(img_dir + random_img).resize(IMAGE_SHAPE)
bird = np.array(bird)/255.0
result = classifier.predict(bird[np.newaxis, ...])
predicted_class = np.argmax(result[0], axis=-1)
print("ID = ", predicted_class)

#birdl.txt is the label you can get on tfhub link given above.
f = list(open("aiy_birds_V1_labelmap.csv").read().splitlines())
# Récupération des labels en latin
latin_labels = [None] * len(f)
for label in f:
    splited = label.split(',')
    id = splited[0]
    if not id.isalpha() :
        latin_labels[int(id)] = splited[1]

imagenet_labels = np.array(latin_labels)
plt.imshow(bird)
plt.axis('off')
predicted_class_name = imagenet_labels[predicted_class]
_ = plt.title("Prediction: " + predicted_class_name.title())
plt.show()

"""
//Converting this model to a tflite model.

converter = tf.lite.TFLiteConverter.from_keras_model(classifier)
classifier.save_weights("..")
tflite_models = converter.convert()
with open('data/model.tflite','wb') as f:
  f.write(tflite_models)

//But it fails to convert.
"""