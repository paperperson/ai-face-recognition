import keras
import cv2
import sys
import gc
import tensorflow as tf
from keras.models import load_model
import numpy as np

img_height = 200
img_width = 200
size = (img_width, img_height)
face_class = ['alvin', 'chae', 'me', 'obama', 'putin', 'trump']
print(face_class[0])

cap = cv2.VideoCapture(2)

data_path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(data_path)
model = load_model('D:/cs final project/me.face.model.h5')


# img = keras.preprocessing.image.load_img(
#     'D:/cw1.jpg', target_size=(200, 200)
# )
# img2 = cv2.imread('D:/alvin0.jpg')
# # gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(img2, 1.3, 5)
# for (x, y, w, h) in faces:
#     cv2.rectangle(img2, (x, y), (x + w, y + w), (0, 255, 0), 2)
#     cropped_image = img2[y - 10: y + h + 10, x - 10: x + w + 10]
# gray = cv2.cvtColor(cropped_image,cv2.COLOR_BGR2GRAY)
# cv2.imshow('chae', cropped_image)
# cv2.waitKey(0)
# # img_array = keras.preprocessing.image.img_to_array(img)
# processed_image = cv2.resize(gray, (200, 200))
# cv2.imshow('chae2', processed_image)
# cv2.waitKey(0)
# img_array = tf.expand_dims(processed_image, 0)
# print(img_array)
# predictions = model.predict(img_array)
# face_id = np.argmax(predictions, axis=1)
# print(face_id)
# print(predictions)

while(1):
    ret, frame = cap.read()
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + w), (0, 255, 0), 2)
            cropped_image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
            try:
                cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                continue
            try:
                processed_image = cv2.resize(cropped_image, (200, 200))
            except Exception as e:
                continue
            processed_image = keras.preprocessing.image.img_to_array(processed_image)
            processed_image = tf.expand_dims(processed_image, 0)
            detected_face = model.predict(processed_image)
            detected_face_id = np.argmax(detected_face, axis=1)
            try:
                cv2.putText(frame, face_class[int(detected_face_id)],
                            (x + 20, y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 0, 255),
                            3)
            except Exception as e:
                continue
            # if detected_face_id == 2:
            #     cv2.putText(frame,'me',
            #                 (x + 20, y + 20),
            #                 cv2.FONT_HERSHEY_SIMPLEX,
            #                 1,
            #                 (255,0,255),
            #                 3)
            # elif detected_face_id == 1:
            #     cv2.putText(frame, 'chae',
            #                 (x + 20, y + 20),
            #                 cv2.FONT_HERSHEY_SIMPLEX,
            #                 1,
            #                 (255,0,255),
            #                 3)
            # elif detected_face_id == 0:
            #     cv2.putText(frame, 'alvin',
            #                 (x + 20, y + 20),
            #                 cv2.FONT_HERSHEY_SIMPLEX,
            #                 1,
            #                 (255, 0, 255),
            #                 3)
            # else:
            #     pass
    cv2.imshow('camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



