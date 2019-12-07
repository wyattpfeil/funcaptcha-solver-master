
import random
import string

import cv2
import imutils
import glob

image = cv2.imread("./TestingImgsComplete/vouhgnnkug110.png")
image = cv2.resize(image, (224, 224))
cv2.imwrite("Resized224-110Deg.png", image)