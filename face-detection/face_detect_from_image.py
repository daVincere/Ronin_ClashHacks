import cv2
import sys

# Gets user specified values
imagePath = sys.argv[1]
cascPath = sys.argv[2]

#  create haar cascade
#  what?
faceCascade = cv2.CascadeClassifier(cascPath)

# The the image for it's image cascPath
image = cv2.imread(imagePath)
# this converts the colors of the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# faceCascade is the internal command of openCV, I guess
# "detectMultiScale" detects the objects in the image
# we are calling it on "faceCascade" hence it will detect only faces
faces = faceCascade.detectMultiScale(  
    gray,
    # scaleFactor compensates for the faces that might seem bigger coz of their close
    # proximity to the camera
    scaleFactor = 1.1,
    # the minumum neighbours near the current detected face
    minNeighbors = 5,
    # minimum size of the face that it detects
    minSize=(30,30),
    # for openCV2 it'd be cv2.cv.CV_HAAR_SCALE_IMAGE
    flags = cv2.CASCADE_SCALE_IMAGE
     )

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)
