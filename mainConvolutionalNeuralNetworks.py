import numpy as np
import pandas as pd
import zipfile
from sklearn import preprocessing
from ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork
from sklearn.model_selection import KFold
from sklearn import metrics
import gc

"""Anomaly Detection using Convolutional Neural Network

This script reads credit card data and implements anomaly detection using Convolutional Neural Network.
The model's performance is measured with KFold Cross Validation, where in every test is included all the
anomaly data, by calculating accuracy, recall, precision and F1 score.

"""
#Initialized parameters
numberOfSnapshots = 50
snapshotShape = (5,6)
threshold =  0.000006
numberOfEpochs = 30
batchSize = 240
kFold = 10
printStatistics = False
semiRandom = True

#Load Data creditcard.csv
zf = zipfile.ZipFile('datasets/creditcard.zip')
data = pd.read_csv(zf.open('creditcard.csv'))

#Distinguish Data (fraud or not)
anomalyData = data[data.Class == 1]
normalData = data[data.Class == 0]

#Delete unnecessary columns
del anomalyData["Class"]
del normalData["Class"]

#Convert Pandas Streams to Numpy arrays
normalArray = np.array(normalData)
anomalyArray = np.array(anomalyData)

#Normalize data
xNormalData = preprocessing.normalize(normalArray, axis = 0)
xAnomalyData = preprocessing.normalize(anomalyArray, axis=0)

#Randomize data
np.random.shuffle(xNormalData)
np.random.shuffle(xAnomalyData)

# k fold cross validation
kf = KFold(n_splits=kFold, shuffle=True)
kIndex = 0
accuracies = 0
recalls = 0
precisions = 0
f1Scores = 0
for trainIndex, testIndex in kf.split(xNormalData):
    xTrain = xNormalData[trainIndex]
    xTest = np.concatenate((xNormalData[testIndex], xAnomalyData), axis=0)
    yTest = np.concatenate((np.zeros(np.size(xNormalData[testIndex], axis=0)), np.ones(np.size(xAnomalyData, axis=0))), axis=0)
    #Create model
    convolutionalNeuralNetwork = ConvolutionalNeuralNetwork(xTrain, snapshotShape=snapshotShape, numberOfSnapshots=numberOfSnapshots, threshold=threshold, statistics=printStatistics, semiRandom=semiRandom)
    convolutionalNeuralNetwork.createCNN(numberOfEpochs=numberOfEpochs, batchSize=batchSize)
    #Make predictions
    predictions = convolutionalNeuralNetwork.predict(xTest)
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
    print(str(kIndex) + " K Fold")
    print("Accuracy: " + str(accuracy * 100) + "%")
    print("Recall: " + str(recall * 100) + "%")
    print("Precision: " + str(precision * 100) + "%")
    print("F1 Score: " + str(f1Score * 100) + "%")
    print()
    #Force the Garbage Collector
    gc.collect()
#Calculate overal metrics
finalAccuracy = accuracies / k
finalRecall = recalls / k
finalPrecision = precisions / k
finalF1Score = f1Scores / k

#Print overal metrics
print("-------------------Overal--------------------------------")
print("Accuracy: " + str(finalAccuracy * 100) + "%")
print("Recall: " + str(finalRecall * 100) + "%")
print("Precision: " + str(finalPrecision * 100) + "%")
print("F1 Score: " + str(finalF1Score * 100) + "%")
print("----------------------------------------------------------")
