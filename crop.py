import os
import cv2

data_path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(data_path)
path_name = 'D:/train_pics/putin/'
def read_path(file_pathname):
    for filename in os.listdir(file_pathname):
        img = cv2.imread(file_pathname + filename)
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + w), (0, 255, 0), 2)
                img_name = path_name + filename
                print(img_name)
                cropped_image = img[y - 10: y + h + 10, x - 10: x + w + 10]
                try:
                    cv2.imwrite(img_name, cropped_image)
                except Exception as e:
                    pass
                continue



read_path('D:/train_pics2/putin/')

# print(os.listdir('D:/train_pics/trump'))
# cv2.imshow('D:/train_pics/trump')