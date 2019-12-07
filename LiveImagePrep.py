# import the necessary packages
import random
import string

import cv2
import imutils
import glob


def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))


SolvedImages = [cv2.imread(img) for img in glob.glob("LiveImageTesting/*.png")]

for image in SolvedImages:
    ImgName = randomString()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
    thresh = thresh[80:195, 93:208]
    EndRotation = 0
    rotatedImage = imutils.rotate(thresh, EndRotation * -1)
    DoneName = ImgName + "-" + str(EndRotation)
    EndPath = "./LiveImageTesting/" + randomString() + "READY" + str(EndRotation) + ".png"
    print(EndPath)
    cv2.imwrite(EndPath, rotatedImage)