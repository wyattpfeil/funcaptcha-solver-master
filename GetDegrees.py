import tensorflow.keras
from PIL import Image
import numpy as np
import random
import string

import cv2
import imutils
import glob
import os

categories = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]

sourceImageFileName = "TestIm4"
sourceImagePath = "./TestingImgsComplete/" + sourceImageFileName + ".png"
threshImagePath = "./TestingImgsComplete/" + sourceImageFileName + "-Thresh.png"
resizedImagePath = "./TestingImgsComplete/" + sourceImageFileName + "-Resized.png"
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('./converted_keras/keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3 ), dtype=np.float32)

def colorToThresh():
    image = cv2.imread(sourceImagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
    thresh = thresh[80:195, 93:208]
    cv2.imwrite(threshImagePath, thresh)

colorToThresh()

image = cv2.imread(threshImagePath)

image = cv2.resize(image, (224, 224))

cv2.imwrite(resizedImagePath, image)


# Replace this with the path to your image
CaptchaImage = Image.open(resizedImagePath)
# Make sure to resize all images to 224, 224 otherwise they won't fit in the array
CaptchaImage = CaptchaImage.resize((224, 224))

image_array = np.asarray(CaptchaImage)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
np.resize(normalized_image_array, (224, 224, 3))
# Load the image into the array
data[0] = normalized_image_array

# run the inference
prediction = model.predict(data)
print(prediction)

confidentIndex = np.argmax(prediction)
print(confidentIndex)
print(categories[confidentIndex])
print(prediction[0][confidentIndex])

os.remove(resizedImagePath)
os.remove(threshImagePath)