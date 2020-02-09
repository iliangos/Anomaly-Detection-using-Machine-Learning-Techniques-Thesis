import numpy as np
import pandas as pd
import zipfile
from sklearn import preprocessing
from sklearn.model_selection import KFold
from playsound import playsound
from sklearn import metrics
from sklearn.svm import OneClassSVM
import gc

"""Anomaly Detection using Support Vector Machines.

This script reads credit card data and implements anomaly detection using Support Vector Machines.
The model's performance is measured with KFold Cross Validation by calculating accuracy, recall,
precision and F1 score.

"""
#Initialized parameters
kFold = 10
sample = False
kernel="rbf"
nu=0.003
gamma='scale'

#Load Data
if sample:
    #Load Data creditcard_sampledata.csv
    dataFileName = "datasets/creditcard_sampledata.csv"
    data = pd.read_csv(dataFileName)
    del data["Unnamed: 0"]
else:
    #Load Data creditcard.csv
    zf = zipfile.ZipFile('datasets/creditcard.zip')
    data = pd.read_csv(zf.open('creditcard.csv'))

#Distinguish Data (normal or anomaly)
anomalyData = data[data.Class == 1]
normalData = data[data.Class == 0]

#Delete unnecessary columns that contains the class information (anomaly of normal)
del anomalyData["Class"]
del normalData["Class"]

#Convert Pandas Streams to Numpy arrays
normalArray = np.array(normalData)
anomalyArray = np.array(anomalyData)

#Scale Data
min_max_scaler = preprocessing.MaxAbsScaler()
xNormalData = min_max_scaler.fit_transform(normalArray)
xAnomalyData = min_max_scaler.fit_transform(anomalyArray)

# #Scale Data
# min_max_scaler = preprocessing.MaxAbsScaler()
# normalArray = min_max_scaler.fit_transform(normalArray)
# anomalyArray = min_max_scaler.fit_transform(anomalyArray)

# # #Normalize data
# xNormalData = preprocessing.normalize(xNormalData, axis = 0)
# xAnomalyData = preprocessing.normalize(xAnomalyData, axis=0)



#Randomize data
np.random.shuffle(xNormalData)
np.random.shuffle(xAnomalyData)

#Initialize overal metrics
kIndex = 0
accuracies = 0
recalls = 0
precisions = 0
f1Scores = 0
#Implement k fold cross validation
kf = KFold(n_splits=kFold, shuffle=True)
for trainIndex, testIndex in kf.split(xNormalData):
    #Training data (normal) for every k
    xTrain = xNormalData[trainIndex]
    #Test data (normal and all anomaly data) for every k
    xTest = np.concatenate((xNormalData[testIndex], xAnomalyData), axis=0)
    yTest = np.concatenate((np.zeros(np.size(xNormalData[testIndex], axis=0)) + 1,-1 * np.ones(np.size(xAnomalyData, axis=0))), axis=0)
    #Create Support Vector Machines model
    svm = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    svm.fit(xTrain)
    #Make predictions
    predictions = svm.predict(xTest)
    #Calculate metrics for every k
    accuracy = metrics.accuracy_score(yTest, predictions)
    recall = metrics.recall_score(yTest, predictions)
    precision = metrics.precision_score(yTest, predictions)
    f1Score = metrics.f1_score(yTest, predictions)
    #Partial calculations of overal metrics
    accuracies += accuracy
    recalls += recall
    precisions += precision
    f1Scores += f1Score
    kIndex += 1
    #Print metrics for every k
    print(str(kIndex) + " Fold Iteration:")
    print("Accuracy: " + str(accuracy * 100) + "%")
    print("Recall: " + str(recall * 100) + "%")
    print("Precision: " + str(precision * 100) + "%")
    print("F1 Score: " + str(f1Score * 100) + "%")
    print()
    #Force the Garbage Collector
    gc.collect()
#Calculate overal metrics
finalAccuracy = accuracies / kFold
finalRecall = recalls / kFold
finalPrecision = precisions / kFold
finalF1Score = f1Scores / kFold

#Print overal metrics
print("-------------------Overal--------------------------------")
print("Accuracy: " + str(finalAccuracy * 100) + "%")
print("Recall: " + str(finalRecall * 100) + "%")
print("Precision: " + str(finalPrecision * 100) + "%")
print("F1 Score: " + str(finalF1Score * 100) + "%")
print("----------------------------------------------------------")
