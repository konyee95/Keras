# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to input dataset of images")
ap.add_argument("-m", "--model", required=True,
    help="path to output trained model")
ap.add_argument("-l", "--label--bin", required=True,
    help="path to output label binarizer")
ap.add_argument("-p", "--plot", required=True,
    help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args['dataset'])))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths
    # load the image, resize the iamge to be 32*32 pixels (ignoring 
    # aspect ratio), flatten the imege into 32x32x3=3072 pixel image
    # into a list, and store the image in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image,(32,32)).flatten()
    data.append(image)

    # extract the class label from the image path and update the 
    # label list
    label = imagePath.split(os.path.step)[-2]
    labels.append(label)

    # scale the raw pixel intensities to the range 
    data = np.array(data, dtype="float")/255.0
    labels = np.array(labels)

    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (trainX, testX, tainY, testY) = train_test_split(data,
        labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors (for 2-class, binary 
# classification you should use Keras to categorical function 
# instead as the scilit-learn's LabelBinarizer willnot return a 
# vector)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# define the 3072-1024-522-3 architecture using keras
model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="sigmoid"))
model.add(Dense(512, activation="sigmoid"))
model.ass(Dense(lb.classes_), activation="softmax")