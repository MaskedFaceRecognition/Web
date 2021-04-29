import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model('models/model_best_0_2.h5')
img = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (48, 48))
'''

img2 = np.array(img)
print(np.array(img2))
img2 = cv2.resize(img2, (48, 48))
print(img2.shape)
img2.reshape(-1, 48, 48)
print(img2.shape)
'''

img = img[..., np.newaxis]
img = img[..., None]
img = img.reshape(-1, 48, 48, 1)
img = tf.reshape(img, (-1, 48, 48, 1))
# img = np.expand_dims(img, axis=-1)
print(img.shape)

# img = cv2.resize(img, (48, 49))
# print(img.shape) # (262, 192)
prediction = model.predict(img)
print(prediction)
