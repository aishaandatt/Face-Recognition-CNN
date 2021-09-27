# Face-Recogniton-CNN

## OverView
This main purpose of this project is to improve the accuracy of facial recogniton for large datasets. We used CNN for training our model unline using Harrcasscade files for detecting and recognising faces, it would not work for large datasets.

## Working

### Taking Images
Consider a scenario where a user wants to register itself, the model will first check if it already exists in the database or not, if yes it will throw a message 'user exists else it will start taking images of that person. For taking images we used harrcascade_frontal_face.xml file where it will detect the face of the person and will take 120 images of only the face which is ideal for CNN becuse this way the dataset will not contain any unnecessary information of the users. The model will assign the ID to the user and stores it's name and ID in the database (mongoDB).

### Training
CNN is used to train images, for demonstration we first used 100 celebrities dataset from kaggle so we had to first preprocess the images, then we used 5 Hidden Layers and BatchNormalistion along with LSTM and Dropout layers. We used 30 Epochs achiving an accuracy of 85%.

## Tracking Images
While authenticating the model first checks if the person exists in the dataset or not, then it matches the name of that person and the ID provided by the user and if all that matches it will Result as "Found" else "Not Found".



