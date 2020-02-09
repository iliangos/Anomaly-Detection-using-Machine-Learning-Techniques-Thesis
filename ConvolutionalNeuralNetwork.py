import numpy as np
from sklearn.metrics import mean_squared_error as mse
from keras import models
from keras import layers
from keras.layers import Convolution2D as Conv2D
from keras.layers import Flatten
from keras.layers import Dense


class ConvolutionalNeuralNetwork:
    """This class implements Convolutional Neural Network for anomaly detection.

    More specific, creates a Convolutional Neural Network that consists of three convolutional layers,
    one flatten layer and one dense layer.

    Parameters
    ----------
    xTrain : numpy array
        This contains the training data.
    snapshotShape : tuple
        Ths contains the shape o snapshot's array
    numberOfSnapshots: integer
        This contains the number of snapshots.
    validationData: numpy array
        This contains the validation data.
    threshold : float
        This is the mininum acceptable mean squared error from predicted and target data.
    semiRandom : boolean
        This defines if the class will use or not random pseudorandom permutations.
    statistics : boolean
        This is the option to printing or not statistics.

    Attributes
    ----------
    xTrain : numpy array
        This contains the training data.
    validationData: numpy array
        This contains the validation data.
    threshold : float
        This is the mininum acceptable mean squared error from predicted and target data.
    statistics : boolean
        This is the option to printing or not statistics.
    convolutionalNetwork : instance of models
        This contains the convolutional neural network.
    snapshotShape : tuple
        Ths contains the shape o snapshot's array
    numberOfSnapshots : integer
        This contains the number of snapshots.
    permutations : ndarray
        This contains the permutation of each snapshot

    """

    def __init__(self, xTrain, snapshotShape=None, numberOfSnapshots=None, validationData=None, threshold=None, semiRandom=False, statistics=False):
        """Implement constructor of ConvolutionalNeuralNetwork class.

        The constructor initialize the attributes of the class ConvolutionalNeuralNetwork.
        """
        self.threshold = threshold
        self.xTrain =  np.array(xTrain)
        self.validationData = np.array(validationData)
        self.convolutionalNetwork = None
        self.statistics = statistics
        self.snapshotShape = snapshotShape
        self.numberOfSnapshots = numberOfSnapshots
        self.permutations = self.createPermutations(semiRandom=semiRandom)

    def createCNN(self, numberOfEpochs=30, batchSize=240, validationSplit=0.001):
        """Create Convolutional Neural Network.

        This function initially transform input data by creating snapshots,
        and then creates and trains the Convolutional Neural Network model
        using this data.

        Parameters
        ----------
        numberOfEpochs : int
            This is the number of epochs in model's training.
        batchSize : int
            The is the batch size.
        validationSplit : float
            This is the validation split value of traning data.

        Attributes
        ----------
        inputShape : tuple
            The shape of input data
        convolutionalNetwork : instance of models class
            The convolutional neural network model.
        yTrain : numpy array
            The target values of training data.
        predictedData : numpy array
            The predicted data.

        """
        #Transform data by creating snapshots
        xTrain = self.transformDataset(self.xTrain)
        #Create Convolutional Neural Network
        inputShape = (self.numberOfSnapshots, self.snapshotShape[0], self.snapshotShape[1])
        convolutionalNetwork = models.Sequential()
        convolutionalNetwork.add(Conv2D(inputShape[0], kernel_size=(2, 2), activation='relu', input_shape=inputShape,data_format="channels_first"))
        convolutionalNetwork.add(Conv2D(inputShape[0], kernel_size=(2, 2), activation='relu',data_format="channels_first"))
        convolutionalNetwork.add(Conv2D(inputShape[0], kernel_size=(2, 2), activation='relu',data_format="channels_first"))
        convolutionalNetwork.add(Flatten())
        convolutionalNetwork.add(Dense(inputShape[0], activation=None))
        #Train the model.
        convolutionalNetwork.compile(loss="mean_squared_error", optimizer="adam", metrics=['accuracy'])
        yTrain = self.createTargetValues(xTrain)
        convolutionalNetwork.fit(xTrain, yTrain, epochs=numberOfEpochs, batch_size=batchSize, validation_split=validationSplit)
        self.convolutionalNetwork = convolutionalNetwork
        if self.statistics:
            #Print Convolutional Neural Network summary.
            convolutionalNetwork.summary()
        if self.threshold == None:
            #Calculate threshold.
            predictedData =  self.convolutionalNetwork.predict(xTrain)
            self.calcThreshold(yTrain, predictedData)

    def predict(self, xData):
        """Predict the class of one or more instances.

        This function choose the candidate class by taking into consideration
        the mean squared error and the threshold.

        Parameters
        ----------
        xData : ndarray
            The instances.

        Attributes
        ----------
        yData : numpy array
            The target's values.
        predictedData : numpy array
            Contains the predicted data.
        finalPredictions : numpy array
            Contains the the final predictions.


        Returns
        -------
        : ndarray
            The final predictions.

        """
        #Create additional data
        xData = self.transformDataset(xData)
        #Create target values
        yData = self.createTargetValues(xData)
        predictedData = self.convolutionalNetwork.predict(xData)
        #Make final predictions
        finalPredictions = np.zeros(yData.shape[0])
        #Choose class
        for index in range(yData.shape[0]):
            if mse(yData[index],predictedData[index],multioutput="raw_values") > self.threshold:
                finalPredictions[index] = 1
        #Print statistics
        if self.statistics:
            self.printStatistics(yData, predictedData, "In Test Data")
        return finalPredictions

    def printStatistics(self, yData, predictedData, position=None):
        """Prints information about the class.

        This method prints the threshold value and the min, mean, and the max
        values of mean squared errors.

        Parameters
        ----------
        yData : numpy array
            The target's values.
        predictedData : numpy array
            Contains the predicted data.
        position : string
            The position of printing values.

        Attributes
        ----------
        yData : numpy array
            The target's values.
        predictedData : numpy array
            Contains the predicted data.

        """
        yData = np.transpose(yData)
        predictedData = np.transpose(predictedData)
        print("-----------------------------")
        if position != None:
            print(str(position))
        print("Threshold: ", str(self.threshold))
        print("Mean Squared Error:")
        print("Max : " + str(np.amax(mse(yData,predictedData, multioutput="raw_values"))))
        print("Mean : " + str(np.mean(mse(yData,predictedData, multioutput="raw_values"))))
        print("Min : " + str(np.min(mse(yData,predictedData, multioutput="raw_values"))))
        print("--------------------------------------")

    def transformDataset(self, xData):
        """Creates snapshots.

        This method creates for every instance of data a number of random snapshots.
         As snapshot is considered a different permutation of instance's values.

        Parameters
        ----------
        xData : ndarray
            The instances.

        Attributes
        ----------
        newData : list
            This list contains all the created snapshots.
        newDataRow : list
            This list contains the created snapshots of one instance.

        Returns
        -------
        : ndarray
            The created snapshots.

        """
        #Create snapshots using the random created permutations
        newData = []
        for dataRow in xData:
            newDataRow = []
            for snapshotIndex in range(self.numberOfSnapshots):
                newDataRow.append(dataRow[self.permutations[snapshotIndex]].reshape(self.snapshotShape))
            newData.append(newDataRow)
        return np.array(newData)

    def createTargetValues(self,xData):
        """Creates the target values for transformed dataset (dataset with snapshots).

        This method creates the target values of the given dataset by calculating the trace
        of every snapshot. The trace of an array is the sum along it's diagonals.

        Parameters
        ----------
        xTrain : numpy array
            The instances.

        Attributes
        ----------
        yData : list
            This list contains all the created target values.
        yDataRow : list
            This list contains the created target values of one instance.

        Returns
        -------
        : numpy array
            The target values.

        """
        yData = []
        for dataRow in xData:
            yDataRow = []
            for channelIndex in range(self.numberOfSnapshots):
                yDataRow.append(np.trace(dataRow[channelIndex]))
            yData.append(yDataRow)
        return np.array(yData)

    def calcThreshold(self, targets, predictions):
        """Calculates the prediction threshold.

        This method calculates the threshold of mean squared error between
        the pedictions and the targets.

        Parameters
        ----------
        targets : numpy array
            The target values.
        predictions : numpy array
            The model's predictions.

        Attributes
        ----------
        mseValues : numpy array
            The mean squared errors between targets and predictions.

        Returns
        -------
        : float
            The max mean squared error of targets and predictions.

        """
        mseValues = mse(np.transpose(targets),np.transpose(predictions),multioutput="raw_values").reshape((targets.shape[0],1))
        self.threshold =  np.max(mseValues).reshape((1,1))

    def createPermutations(self, semiRandom=False):
        """Creates random permutations for snapshots.

        This method creates random or pseudorandom permutaions
        that are used to create snapshots.

        Parameters
        ----------
        semiRandom : boolean
            This defines if the permutations will be random or pseudorandom.

        Attributes
        ----------
        permutations : list
            This list contains the permutations.

        Returns
        -------
        : ndarray
            The perrmutations.

        """
        permutations = []
        for index in range(self.numberOfSnapshots):
            if semiRandom:
                np.random.seed(index)
            permutations.append(np.random.permutation(self.xTrain.shape[1]))
        return np.array(permutations)
