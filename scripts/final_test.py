#-----------------------------------
# TRAINING OUR MODEL
#-----------------------------------

# import the necessary packages
import h5py
import numpy as np
import pickle
import os
import glob
import cv2
import joblib
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.externals import joblib

import urllib3
from urllib import urlopen
import re
import os


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

import mahotas

# feature-descriptor-1:
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

num_trees=200
test_size=0.1
seed=9
bins = 8
train_path = "dataset/train"
test_path="dataset/test"


def get_image():
    print("download starting")
    img_data=urlopen('https://firebasestorage.googleapis.com/v0/b/picfi-79b51.appspot.com/o/image.jpg?alt=media&token=5944cfb2-6e3a-4861-a0e3-8e05298ea787').read()
    filename = "dataset/test/1.jpg"
    filename2= "static/1.jpg"
    with open(filename, 'wb') as f:
        f.write(img_data)
    with open(filename2, 'wb') as f2:
        f2.write(img_data)
        print("download completed")
        return("OK! downloaded ")
   


def perdict(complete_path):
    #get_image()
    #train_labels = os.listdir(train_path)
   # train_labels= ['bluebell', 'buttercup', 'coltsfoot', 'cowslip', 'crocus', 'daffodil', 'daisy', 'dandelion', 'fritillary', 'iris', 'lilyvalley', 'pansy', 'snowdrop', 'sunflower', 'tigerlily', 'tulip', 'windflower']
    train_labels = ['c_0', 'c_1', 'c_10', 'c_11', 'c_12', 'c_13', 'c_14', 'c_15', 'c_16', 'c_17', 'c_18', 'c_19', 'c_2', 'c_20', 'c_21', 'c_23', 'c_24', 'c_25', 'c_26', 'c_27', 'c_28', 'c_29', 'c_3', 'c_30', 'c_31', 'c_32', 'c_33', 'c_34', 'c_35', 'c_36', 'c_37', 'c_4', 'c_5', 'c_6', 'c_7', 'c_8', 'c_9']

    """
    # import the feature vector and trained labels
    h5f_data = h5py.File('output/data.h5', 'r')
    h5f_label = h5py.File('output/labels.h5', 'r')

    global_features_string = h5f_data['dataset_1']
    global_labels_string = h5f_label['dataset_1']

    global_features = np.array(global_features_string)
    global_labels = np.array(global_labels_string)

    h5f_data.close()
    h5f_label.close()

    # verify the shape of the feature vector and labels
    print("[STATUS] features shape: {}".format(global_features.shape))
    print("[STATUS] labels shape: {}".format(global_labels.shape))

    print("[STATUS] training started...")

    # split the training and testing data
   
    (trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                              np.array(global_labels),
                                                                                              test_size=test_size,
                                                                                             random_state=seed)
    """
    # filter all the warnings
    import warnings
    warnings.filterwarnings('ignore')

    #-----------------------------------
    # TESTING OUR MODEL
    #-----------------------------------

    # to visualize results
    import matplotlib.pyplot as plt
    filename = 'finalized_model.sav'
    # create the model - Random Forests
    """
    clf  = RandomForestClassifier(n_estimators=100, random_state=9)

    # fit the training data to the model
    #clf.fit(trainDataGlobal, trainLabelsGlobal)
    clf.fit(np.array(global_features),np.array(global_labels))

    #save the model
   
    pickle.dump(clf, open(filename, 'wb'))
"""
    # load the model from disk
    print("loading model")
    clf = pickle.load(open(filename, 'rb'))
    # path to test data
    #test_path = "dataset/test"
    test_path = complete_path
    # loop through the test images
    print("inside final_test")
    for file in glob.glob(test_path):
        # read the image
        print(file)
        image = cv2.imread(file)

        # resize the image
       # image = cv2.resize(image, fixed_size)

        ####################################
        # Global Feature extraction
        ####################################
        fv_hu_moments = fd_hu_moments(image)
        fv_haralick   = fd_haralick(image)
        fv_histogram  = fd_histogram(image)

        ###################################
        # Concatenate global features
        ###################################
        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

        # predict label of test image
        prediction = clf.predict(global_feature.reshape(1,-1))[0]
        return(train_labels[prediction])
    
       


