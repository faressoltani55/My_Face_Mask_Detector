import warnings
import numpy as np
import cv2
from tensorflow.keras.models import load_model

warnings.filterwarnings('ignore')
cvNet = cv2.dnn.readNetFromCaffe('Caffe_face_detection/architecture.txt','Caffe_face_detection/weights.caffemodel')
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

threshold = 0.90
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
font = cv2.FONT_HERSHEY_COMPLEX
model = load_model('Face_Mask_Detection_model.h5')


def preprocessing(img):
    img = img.astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)])
    return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))


gamma = 2.0
rows = 3
cols = 2
axes = []
assign = {'0':'With Mask','1':"Without Mask"}
img_size = 124
while True:
    success, imgOrignal = cap.read()
    faces = facedetect.detectMultiScale(imgOrignal, 1.3, 5)
    for x, y, w, h in faces:
        cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(50,50,255),2)
        cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (50,50,255),-2)
        crop_img = imgOrignal[y:y + h, x:x + h]
        image = cv2.resize(crop_img, (32, 32))
        image = adjust_gamma(image, gamma=gamma)
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        cvNet.setInput(blob)
        detections = cvNet.forward()
        for i in range(0, detections.shape[2]):
            try:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                frame = image[startY:endY, startX:endX]
                confidence = detections[0, 0, i, 2]
                if confidence > 0.2:
                    im = cv2.resize(frame, (img_size, img_size))
                    im = np.array(im) / 255.0
                    im = im.reshape(1, 124, 124, 3)
                    result = model.predict(im)
                    if result < 0.5:
                        cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.rectangle(imgOrignal, (x, y - 40), (x + w, y), (0, 255, 0), -2)
                        cv2.putText(imgOrignal, "Mask", (x, y - 10), font, 0.75,
                                    (255, 255, 255), 1, cv2.LINE_AA)
                    else:
                        cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (50, 50, 255), 2)
                        cv2.rectangle(imgOrignal, (x, y - 40), (x + w, y), (50, 50, 255), -2)
                        cv2.putText(imgOrignal, "No Mask", (x, y - 10), font, 0.75,
                                    (255, 255, 255), 1, cv2.LINE_AA)

            except:
                pass
    cv2.imshow("WindowFrame", imgOrignal)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
