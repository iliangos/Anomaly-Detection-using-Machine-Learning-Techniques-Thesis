import numpy as np
from keras import models
from keras import layers
from sklearn.metrics import mean_squared_error as mse

class DeepAutoencoder:
    """This class implements Deep Autoencoder for anomaly detection.

    More specific, creates a Deep Autoencoder where the number of neuron defined by
    the given compression ratio. The predictions are made by calculating the mean
    squared error from predicted and target data.

    Parameters
    ----------
    xTrain : numpy array
        This contains the training data.
    validationData: numpy array
        This contains the validation data.
    threshold : float
        This is the mininum acceptable mean squared error from predicted and target data.

    Attributes
    ----------
    trainData : numpy array
        This contains the training data.
    validationData: numpy array
        This contains the validation data.
    threshold : float
        This is the mininum acceptable mean squared error from predicted and target data.
    autoencoder : instance of models
        This contains the deep autoencoder.

    """

    def __init__(self, trainData, validationData=None, threshold=None):
        """Implement constructor of Autoencoder class.

        The constructor initialize the attributes of the class Autoencoder.
        """
        self.threshold = threshold
        self.trainData = np.array(trainData)
        self.validationData = np.array(validationData)
        self.autoencoder = None

    def createAutoencoder(self, compressionRatio, numberOfHiddenLayers=4, numberOfEpochs=200, batchSize=35, printAutoencoderSummary=False):
        """Creates the Autoencoder.

        This function creates and trains the autoencoder. More spesific creates an
        autoencoder with given number of neurons per layer is considered by the compression
        ratio.

        Parameters
        ----------
        compressionRatio : float
            The compression ratio of autoencoder.
        numberOfHiddenLayers : int
            The number of hidden layers.
        numberOfEpochs : int
            This is the number of epochs in model's training.
        batchSize : int
            The number of batchSize in model's training.
        printAutoencoderSummary : boolean
            The flag to print autoencoder's summary information.


        Attributes
        ----------
        compressionRatioPerLayer : float
            The compression ratio of layers.
        autoencoder : instance of models class
            The deep autoencoder model.
        inputSize : int
            The size of input data in axis 1.
        neurons : int
            The number of neurons per layer.
        neuronsPerLayer : list
            Contains the neurons of every layer.

        """
        #Create autoencoder
        compressionRatioPerLayer = np.power(compressionRatio,(1 / (int(numberOfHiddenLayers / 2) + 1)))
        autoencoder = models.Sequential()
        inputSize = np.size(self.trainData, axis=1)
        neurons = int(np.rint(inputSize * compressionRatioPerLayer))
        neuronsPerLayer = []
        autoencoder.add(layers.Dense(neurons,activation="relu",input_shape=(inputSize,)))
        for index in range(0, int(np.round((numberOfHiddenLayers + 1) / 2))):
            neuronsPerLayer.append(neurons)
            neurons = int(np.rint(neurons * compressionRatioPerLayer))
        neuronsPerLayer = np.array(neuronsPerLayer)
        neuronsPerLayer = np.concatenate((neuronsPerLayer,np.flip(neuronsPerLayer[:-1])),axis=0)
        for neurons in neuronsPerLayer[1:]:
            autoencoder.add(layers.Dense(neurons,activation="relu"))
        autoencoder.add(layers.Dense(inputSize,activation=None))
        #Print autoencoder summary
        if printAutoencoderSummary:
            autoencoder.summary()
        #Train autoencoder
        autoencoder.compile(optimizer="adam", loss='mean_squared_error')
        autoencoder.fit(self.trainData, self.trainData, epochs=numberOfEpochs, batch_size=batchSize, shuffle=True)
        self.autoencoder = autoencoder
        if self.threshold == None:
            self.__calcThreshold(self.trainData, self.autoencoder.predict(self.trainData))

    def __calcThreshold(self, targets, predictions):
        """Calculates the prediction threshold.

        This method calculates the threshold of mean squared error between
        the pedictions and the targets.

        Parameters
        ----------
        targets : numpy array
            The target values.
        predictions : numpy array
            The model's predictions.

        Returns
        -------
        : float
            The max mean squared error of targets and predictions.

        """
        self.threshold = np.max(mse(targets,predictions, multioutput="raw_values"))

    def predict(self, data):
        """Predict the class of one or more instances.

        This function choose the candidate class by taking into consideration
        the mean squared error and the threshold.

        Parameters
        ----------
        data : ndarray
            The instances.

        Attributes
        ----------
        predictedData : numpy array
            Contains the predicted data.
        finalPredictions : numpy array
            Contains the the final predictions.


        Returns
        -------
        : numpy array
            The final predictions.

        """
        predictedData = self.autoencoder.predict(data)
        finalPredictions = np.zeros((np.size(data, axis=0), 1))
        for index in range(np.size(predictedData,axis=0)):
            if mse(data[index],predictedData[index],multioutput="raw_values") > self.threshold:
                finalPredictions[index] = 1
        return finalPredictions
