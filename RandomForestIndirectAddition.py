import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing

class RandomForestIndirectAddition:
    """This class implements the Brute Force Random Forest.

    More specific, creates a Random Forest that contains Decision Trees where each tree
    is trained with all avaliable anomaly data and a part of the normal data. The normal
    data has the same size with anomaly data.

    Parameters
    ----------
    xData : numpy array
        This contains the training data.
    yData : numpy array
        This contains the classes of training data.
    maxNumberOfSplits : int
        This is the maximum number of Decition Trees.

    Attributes
    ----------
    xData : numpy array
        This contains the training data.
    yData : int
        This contains the classes of training data.
    maxNumberOfSplits : int
        This is the maximum number of Decition Trees.

    """

    def __init__(self, xData, yData, maxNumberOfSplits=None):
        """Implement constructor of RandomForestIndirectAddition class.

        The constructor initialize the attributes of the class RandomForestIndirectAddition.
        """
        self.xData = np.array(xData)
        self.yData = np.array(yData)
        self.maxNumberOfSplits = maxNumberOfSplits
        self.randomForest = []

    def createRandomForest(self):
        """Create Random Forest.

        This function creates the Random Forest by creating Decision Trees that are
        trained with all avaliable anomaly data and a part of the normal data.

        Attributes
        ----------
        normalData : numpy array
            The normal instances.
        anomalyData : numpy array
            The anomaly instances.
        numberOfSplits : int
            The maximum possible number of Decision Trees.
        x_train : numpy array
            The training data.
        y_train : The training data's targets.
        decisionTreeClassifier : DecisionTreeClassifier class
            The Decition Tree model.

        """
        normalData = self.xData[np.where(self.yData == 0)]
        anomalyData = self.xData[np.where(self.yData == 1)]
        numberOfSplits = int(np.size(normalData) // np.size(anomalyData))
        if self.maxNumberOfSplits != None and self.maxNumberOfSplits < numberOfSplits :
            numberOfSplits = self.maxNumberOfSplits
        for splitedData in np.split(normalData[:(numberOfSplits * np.size(anomalyData, axis=0))],numberOfSplits):
            x_train = np.concatenate((splitedData, anomalyData), axis=0)
            y_train = np.concatenate((np.zeros(np.size(splitedData,axis=0)), np.ones(np.size(anomalyData,axis=0))), axis=0)
            decisionTreeClassifier = DecisionTreeClassifier()
            decisionTreeClassifier.fit(x_train, y_train)
            self.randomForest.append(decisionTreeClassifier)

    def predict(self, data):
        """Predict the class of  one or more instances.

        This function choose the candidate class with the most votes.

        Parameters
        ----------
        data : ndarray
            The instances.

        Attributes
        ----------
        finalPredictions : list of ints
            Contains the the final predictions.
        predictedData : list of ints
            Contains the partly predictions.

        Returns
        -------
        : list of ints
            The final predictions.

        """
        finalPredictions = []
        for instance in data:
            predictedData = []
            for decisionTreeClassifier in self.randomForest:
                predictedData.append(decisionTreeClassifier.predict(np.array([instance])))
            if np.size(predictedData) - np.sum(predictedData) > np.sum(predictedData) :
                finalPredictions.append(0)
            else:
                finalPredictions.append(1)
        return finalPredictions
