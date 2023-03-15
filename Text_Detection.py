import keras_ocr
import cv2
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

pipeline = keras_ocr.pipeline.Pipeline()

image = cv2.imread('photo.jpg')

prediction_groups = pipeline.recognize([image])

for prediction in prediction_groups[0]:
    print(prediction[0])
