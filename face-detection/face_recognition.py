import sys
import cv2, os
import numpy as np
from PIL import Image

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath) 

# recognizer has fns like FaceRecognizer.train and FaceRecognizer.pridict
# 3 face recognizers
# Eigen, Fisher and LBPH Face Regognizer
# recognizer = cv2.createLBPHFaceRecognizer()  for OpenCV2
recognizer = cv2.face.createLBPHFaceRecognizer()

def get_images_and_labels(path):
    image_path = [os.path.join(path, f)for f in os.listdir(path) if not f.endswith('.sad')]

    images = []
    labels = []

    for image_path in image_paths:
        # reads the image and converts to grayscale
        # alternative
        # image_pil = cv2.imread(image_path)
        # gary = cv2.cvtColor(image_pil, cv2.COLOR_RGB2GRAY)
        # but we can't use "imread" as OpenCV doesn't support GIF images
        image_pil = Image.open(image_path).convert('L')
        image = np.array(image_pil, 'uint8')
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))

        faces = faceCascade.detectMultiScale(image)

        for (x,y,w,h) in faces:
            # 
            images.append(image[y: y+h, x:x+w])
            # adds the mood of the persion to the name
            labels.append(nbr)
            cv2.imshow("Adding faces for trainig:", images)
            # which key?
            cv2.waitKey(50)
    return images, labels

# "path" for the training set
path = 'yalefaces'
images, labels = get_images_and_labels(path)
cv2.destroyAllWindows()


#  Now we'll train the camera for accuracy
path = 'yalefaces'
images, labels = get_images_and_labels(path)
cv2.destroyAllWindows()

recognizer.train(images, np.array(labels))

image_paths = [os.path.join(path, f) for f in os.listdir(path)]

for image_path in image_paths:
    # why not do cv2.imread()??
    # and convert to grayscale by 
    # cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    predict_image_pil = Image.open(image_path).convert('L')
    # what?
    # we are taking the image and doing what?
    predict_image = np.array(predict_image_pil, 'uint8')

    faces = faceCascade.detectMultiScale(predict_image)

    for (x, y, w, h) in faces:
        # why these two lines below
        nbr_predicted, conf = recognizer.predict(predict_image[y:y+h, x:x+w])
        nbr_actual = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        
        if nbr_actual == nbr_predicted:
            print "{} is Correctly Recognized with confidence {}".format(nbr_actual, conf)
        else:
            print "{} is Incorrectly Recognized as {}".format(nbr_actual, nbr_predicted)
            cv2.imshow("Recognizing Face", predict_image[y: y + h, x: x + w])
            cv2.waitKey(1000) 