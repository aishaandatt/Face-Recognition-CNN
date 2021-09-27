from pymongo.settings import TopologySettings
from datetime import datetime, timedelta
from pymongo import MongoClient
import tkinter as tk
import cv2
import os
import numpy as np
from PIL import Image
#import pandas as pd
import uuid
import tensorflow as tf
#import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import matplotlib.pyplot as plt
from tensorflow.keras.layers import BatchNormalization
from keras_preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


client = MongoClient('localhost', 27017)
db = client['teat_animal']
collection = db['test_animal']


window = tk.Tk()
window.title("Biometric System")
window.geometry('1400x720')
window.configure(background='#6F4C5B')
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)
message = tk.Label(text="Bio Metric System",
                   fg="blue", font=('roboto', 30, 'bold'))
message.place(x=600, y=20)
lbl2 = tk.Label(window, text="Enter Name", width=20, fg="black",
                bg="#DEBA9D", height=2, font=('roboto', 15, ' bold '))
lbl2.place(x=400, y=300)
txt2 = tk.Entry(window, width=20, bg="#DEBA9D",
                fg="black", font=('roboto', 15, ' bold '))
txt2.place(x=700, y=315)
lbl3 = tk.Label(window, text="Notification : ", width=20, fg="black",
                bg="#DEBA9D", height=2, font=('roboto', 15, ' bold underline '))
lbl3.place(x=400, y=400)
message = tk.Label(window, text="", bg="#DEBA9D", fg="black", width=30,
                   height=2, activebackground="#DEBA9D", font=('roboto', 15, ' bold '))
message.place(x=700, y=400)


def clear():
    txt2.delete(0, 'end')
    res = ""
    message.configure(text=res)


def id():
    k = uuid.uuid1().int << 1
    k1 = int(str(k)[:5])
    print(k1)
    check = collection.find_one({"ID": k1})
    if(check):
        k1 = id()
    return (str)(k1)


def some_function():
    end_time = datetime.now() + timedelta(seconds=10)
    while datetime.now() < end_time:
        TrackImages()


request_number = 0


def folder():
    request_number = TakeImages.Id
    #_dir = ""
    # base dir

    # create dynamic name, like "D:\Current Download\Attachment82673"
    _dir = os.path.join(
        "/Users/aishaandatt/Downloads/Face-Recognition/TrainingImage", 'Class%s' % request_number)

    # create 'dynamic' dir, if it does not exist
    if not os.path.exists(_dir):
        os.makedirs(_dir)


def TakeImages():
    Id = id()
    TakeImages.Id = Id
    name = (txt2.get())
    TakeImages.name = name
    if(name.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 8)
        ka = TrackImages()
        print(ka)
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 8)
            if(ka == 0):
                break

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                # incrementing sample number
                sampleNum = sampleNum+1
                # folder()
                # saving the captured face in the dataset folder TrainingImage
                request_number = Id
                _dir = os.path.join(
                    "/Users/aishaandatt/Downloads/Face-Recognition/TrainingImage/" + "Class%s" % request_number+'/')
                if not os.path.exists(_dir):
                    os.makedirs(_dir)
                cv2.imwrite(_dir + name + "."+Id + '.' +
                            str(sampleNum) + ".jpg", gray[y:y+h, x:x+w])
                # display the frame
                cv2.imshow('frame', img)
            # wait for 100 miliseconds
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum > 120:
                break
        cam.release()
        cv2.destroyAllWindows()
        res = "ID : " + Id + " Name : " + name
        message.configure(text=res)
    dict1 = [{"Name": name, "ID": (int)(Id)}]
    t = 1
    #p = ka
    while(t > 0):
        collection.insert(dict1)
        t = t-1
    print('ID is : ', Id)


def TrainImages():
    train_dir = "/Users/aishaandatt/Downloads/Face-Recognition/TrainingImage"
    generator = ImageDataGenerator()
    train_ds = generator.flow_from_directory(
        train_dir, target_size=(224, 224), batch_size=32)
    classes = list(train_ds.class_indices.keys())
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(96, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.3))
    model.add(Dense(len(classes), activation='softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=["accuracy"])
    model.summary()
    history = model.fit(train_ds, epochs=3, batch_size=32)
    model.save('Human_improved.h5')
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['loss'])
    plt.xlabel('Time')
    plt.legend(['accuracy', 'loss'])
    plt.show()


def getImagesAndLabels(path):
    # get the path of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # print(imagePaths)
    # create empth face list
    faces = []
    # create empty ID list
    Ids = []
    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        pilImage = Image.open(imagePath).convert('L')
        # Now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage, 'uint8')
        # getting the Id from the image
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids


# def TrackImages():
#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     recognizer.read(
#         "/Users/aishaandatt/Downloads/Face-Recognition/TrainingImageLabel/Trainner.yml")
#     harcascadePath = "haarcascade_frontalface_default.xml"
#     faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + harcascadePath)
#     cam = cv2.VideoCapture(0)
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     t = 1
#     end_time = datetime.now() + timedelta(seconds=20)
#     while datetime.now() < end_time:
#         ret, im = cam.read()
#         gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#         faces = faceCascade.detectMultiScale(gray, 1.1, 5)
#         k = 1
#         for(x, y, w, h) in faces:
#             cv2.rectangle(im, (x, y), (x+w, y+h), (225, 0, 0), 2)
#             Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
#             if(conf < 30):
#                 k = 0
#             else:
#                 cv2.rectangle(im, (x, y), (x+w, y+h), (0, 0, 255), 2)
#                 Id = 'Unknown'
#             if(conf > 75):
#                 noOfFile = len(os.listdir("ImagesUnknown"))+1
#                 cv2.imwrite("ImagesUnknown/Image"+str(noOfFile) +
#                             ".jpg", im[y:y+h, x:x+w])
#             if(k == 0):
#                 op = collection.find_one({"ID": Id}, {'_id': False})
#                 acc = "{:0.2f}%".format(100-conf)
#                 cv2.putText(im, (str)(op)+(str)(acc), (x, y+h),
#                             font, 1, (255, 255, 255), 2)
#             else:
#                 cv2.putText(im, (str)(Id), (x, y+h),
#                             font, 1, (255, 255, 255), 2)
#             while(t > 0):
#                 print('ID is', Id)
#                 print('conf is', conf)
#                 if(conf < 30):
#                     y = collection.find_one({"ID": Id})
#                     print(y)
#                     if(y):
#                         print("Found")
#                         ktr = 1
#                         TrackImages.ktr = ktr
#                 else:
#                     print("Not Found")
#                     ktr = 0
#                     TrackImages.ktr = ktr
#                 t = t-1
#         cv2.imshow('im', im)
#         if (cv2.waitKey(1) == ord('q')):
#             break
#     cam.release()
#     cv2.destroyAllWindows()
#     if(y):
#         return 1
#     return 0
train_dir = "/Users/aishaandatt/Downloads/IBM/Celebs_Mega"
generator = ImageDataGenerator()
train_ds = generator.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32)
classes = list(train_ds.class_indices.keys())


def TrackImages():
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + harcascadePath)
    model = load_model(
        '/Users/aishaandatt/Downloads/Face-Recognition/Human_improved.h5')
    video = cv2.VideoCapture(0)
    end_time = datetime.now() + timedelta(seconds=50)
    while datetime.now() < end_time:
        success, frame = video.read()
        cv2.imwrite("image.jpg", frame)
        img = image.load_img("image.jpg", target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        pred = np.argmax(model.predict(x))
        cv2.putText(frame, "predicted  class = " + str(pred), (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        pred = model.predict_classes(x)
        y_pred = model.predict(x)
        k = y_pred[0][pred]
        print(k)
        cv2.imshow("image", frame)
        if cv2.waitKey(1) & 0xFF == ord('a'):
            break


clearButton2 = tk.Button(window, text="Clear", command=clear, fg="black", bg="#DEBA9D",
                         width=20, height=2, activebackground="#DEBA9D", font=('roboto', 15, ' bold '))
clearButton2.place(x=950, y=300)
takeImg = tk.Button(window, text="Register", command=TakeImages, fg="black", bg="#DEBA9D",
                    width=20, height=3, activebackground="#DEBA9D", font=('roboto', 15, ' bold '))
takeImg.place(x=200, y=500)
trainImg = tk.Button(window, text="Train", command=TrainImages, fg="black",
                     bg="#DEBA9D", width=20, height=3, activebackground="black", font=('roboto', 15, ' bold '))
trainImg.place(x=500, y=500)
trackImg = tk.Button(window, text="Authenticate", command=TrackImages, fg="black",
                     bg="#DEBA9D", width=20, height=3, activebackground="black", font=('roboto', 15, ' bold '))
trackImg.place(x=800, y=500)
quitWindow = tk.Button(window, text="Quit", command=window.destroy, fg="black", bg="#DEBA9D",
                       width=20, height=3, activebackground="#DEBA9D", font=('roboto', 15, ' bold '))
quitWindow.place(x=1100, y=500)
window.mainloop()


# LSTM try karo
# Clustering try
