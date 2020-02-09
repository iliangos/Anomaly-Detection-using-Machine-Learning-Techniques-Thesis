import numpy as np
import pandas as pd
import zipfile
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from imblearn.over_sampling import SMOTE
import gc

"""Anomaly Detection using RandomForest and SMOTE over sampling.

This script reads credit card data and implements anomaly detection using RandomForest and SMOTE over sampling.
The model's performance is measured with KFold Cross Validation by calculating accuracy, recall, precision and
F1 score. The KFold Cross Validation is implementing at the same time at both anomaly and normal data.

"""
#Initialized parameters
kFold = 10
nEstimators = 100
minImpurityDecrease = 0

#Load Data
zf = zipfile.ZipFile('datasets/creditcard.zip')
data = pd.read_csv(zf.open('creditcard.csv'))

#Distinguish Data (anomaly or normal)
anomalyData = data[data.Class == 1]
normalData = data[data.Class == 0]

#Delete unnecessary columns that contains the class information (anomaly of normal)
del anomalyData["Class"]
del normalData["Class"]

#Convert Pandas Streams to Numpy arrays
normalArrayUnnormalized = np.array(normalData)
anomalyArrayUnnormalized = np.array(anomalyData)

#Scale Data
min_max_scaler = preprocessing.MaxAbsScaler()
normalArray = min_max_scaler.fit_transform(normalArrayUnnormalized)
anomalyArray = min_max_scaler.fit_transform(anomalyArrayUnnormalized)

# #Normalize data
normalArray = preprocessing.normalize(normalArray, axis = 0)
anomalyArray = preprocessing.normalize(anomalyArray, axis=0)

#Randomize data
np.random.shuffle(normalArray)
np.random.shuffle(anomalyArray)

#Initialize overal metrics
kIndex = 0
accuracies = 0
recalls = 0
precisions = 0
f1Scores = 0
#Implement k fold cross validation in normal and anomaly data at the same time
kf = KFold(n_splits=kFold, shuffle=True)
for normalIndexes, anomalyIndexes in zip(kf.split(normalArray), kf.split(anomalyArray)):
    #Indexes of normal data split
    normalTrainIndex = normalIndexes[0]
    normalTestIndex = normalIndexes[1]
    #Indexes of anomaly data split
    anomalyTrainIndex = anomalyIndexes[0]
    anomalyTestIndex = anomalyIndexes[1]
    #Concatenation of normal and anomaly data
    x_train =  np.concatenate((normalArray[normalTrainIndex], anomalyArray[anomalyTrainIndex]), axis=0)
    y_train =  np.concatenate((np.zeros(np.size(normalTrainIndex)), np.ones(np.size(anomalyTrainIndex))), axis=0)
    x_test =  np.concatenate((normalArray[normalTestIndex], anomalyArray[anomalyTestIndex]), axis=0)
    y_test =  np.concatenate((np.zeros(np.size(normalTestIndex)), np.ones(np.size(anomalyTestIndex))), axis=0)
    #SMOTE over sampling
    smt = SMOTE()
    x_train, y_train = smt.fit_sample(x_train, y_train)
    #Create Random Forest model
    randomForest = RandomForestClassifier(n_estimators =nEstimators, min_impurity_decrease=minImpurityDecrease, random_state=0)
    randomForest.fit(x_train, y_train);
    #Make predictions
    predictions = randomForest.predict(x_test)
    #Calculate metrics for every k
    accuracy = metrics.accuracy_score(y_test, predictions)
    recall = metrics.recall_score(y_test, predictions)
    precision = metrics.precision_score(y_test, predictions)
    f1Score = metrics.f1_score(y_test, predictions)
    #Partial calculations of overal metrics
    accuracies += accuracy
    recalls += recall
    precisions += precision
    f1Scores += f1Score
    kIndex += 1
    #Print metrics for every k
    print(str(kIndex) + " K Fold:")
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
