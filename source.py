import cv2
import sys

imagepath = sys.argv[1]
cascpath = sys.argv[2]

facecascade = cv2.CascadeClassifier(cascpath)


img = cv2.imread('2.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = facecascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30,30)
)

print(f'Foundc{len(faces)} faces!')

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w, y+h),(0,255,0),2 )


cv2.imshow("Faces Found",img)
cv2.waitKey(0)