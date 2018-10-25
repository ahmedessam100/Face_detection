##Face Recognition

#Importing the libraries
import cv2

#Loading the Cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

#defining a function to detect ,input : gray is the image in black and white , frame is the original
#image
def detect(gray, frame):
    scale_factor = 1.2
    #faces is tuples of x,y,w,h where x and h are the co-ordinates of the upper left corner of the rectangeles that will/
    #the face
    #w is the width of the image , h is the height of the image
    faces = face_cascade.detectMultiScale(gray, scale_factor, 5)
    for (x, y, w, h) in faces:
        #draw the rectangle
        #first arg: the image,second arg:upper left corner, third arg:lower right corner
        #fourth arg: color of the rect. , fifth:the thickness of the rect.
        cv2.rectangle(frame, (x, y), (x+w, y+h), (200, 30, 100), 2)
        #region of interest for the gray pic
        roi_gray = gray[y:y+h, x:x+w]
        #region of interest for the original color image
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            # draw the rectangle
            # first arg: the image,second arg:upper left corner, third arg:lower right corner
            # fourth arg: color of the rect. , fifth:the thickness of the rect.
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 21)
        for (sx, sy, sw, sh) in smiles:
            # draw the rectangle
            # first arg: the image,second arg:upper left corner, third arg:lower right corner
            # fourth arg: color of the rect. , fifth:the thickness of the rect.
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
    #returning the original frame with rect. around eyes and faces
    return frame

#Doing some face recognition with the webcam
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas =  detect(gray, frame)
    cv2.imshow('Video', canvas)
    #stop when pressed 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#turn off the webcam
video_capture.release()
#destroy the window in which all images is displayed
cv2.destroyAllWindows()
