import cv2
import sys

cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

# We capture the video by the VideoCapture fn of openCV
# the source of the video is the webcam, hence 0
# we can also pass a video here
video_capture = cv2.VideoCapture(0)

while True:
    # reads one frame a loop and returns a code (idk which one)
    ret, frame = video_capture.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray, 
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (30,30),
        flags = cv2.CASCADE_SCALE_IMAGE
    ) 

for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Dekho Gadha", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# as you can read
video_capture.release()
cv2.destroyAllWindows()
