import tensorflow as tf
import argparse
# Define model and output directory arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Folder that the saved model is located in',
                    default='exported-models/my_tflite_model/saved_model')
parser.add_argument('--output', help='Folder that the tflite model will be written to',
                    default='exported-models/my_tflite_model')
args = parser.parse_args()

converter = tf.lite.TFLiteConverter.from_saved_model(args.model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

tflite_model = converter.convert()

output = args.output + '/model.tflite'
with tf.io.gfile.GFile(output, 'wb') as f:
  f.write(tflite_model)