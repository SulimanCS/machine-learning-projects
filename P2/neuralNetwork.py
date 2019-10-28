# ==============================================
# Name: Suliman Alsarraf
# Assignment: Project 2
# Class: CS445 Machine Learning
# ==============================================

import sys
import os
import csv
import time
import numpy as np
import math
import random
import pandas as pd

# record when the program has started
start_time = time.time()

def setupTrainSet():

	# get the training set path relative to the
	# execution path
	TRAINSET = 'mnist_train.csv'
	dirr = os.path.dirname(__file__)
	path = os.path.join(dirr, TRAINSET)

	# initialize the lables/output and pixels lists
	# the lables are meant to be 1D list, and the pixels 2D
	labels = []
	pixels = []

	# read in the lables and images info
	with open(path, 'r') as fil:
		r = csv.reader(fil)
		# preprocessing, go through every pixel of every image
		for i, line in enumerate(r):
			lbl = (int(line[0]))
			pxl = line[1:]
			# insert the bias at the beginning of the list
			pxl.insert(0, 1.0)

			# divide every pixel value to be between 0 and 1
			for j in range(len(pxl)):
				pxl[j] = float(pxl[j])
				pxl[j] = pxl[j]/255

			# store the values to be returned
			labels.append(lbl)
			pixels.append(pxl)

	return labels, pixels

def setupTestSet():

	# get the testing set path relative to the
	# execution path
	TESTSET = 'mnist_test.csv'
	dirr = os.path.dirname(__file__)
	path = os.path.join(dirr, TESTSET)

	# initialize the lables/output and pixels lists
	# the lables are meant to be 1D list, and the pixels 2D
	labels = []
	pixels = []

	# read in the lables and images info
	with open(path, 'r') as fil:
		r = csv.reader(fil)
		# preprocessing, go through every pixel of every image
		for i, line in enumerate(r):
			lbl = (int(line[0]))
			pxl = line[1:]
			# insert the bias at the beginning of the list
			pxl.insert(0, 1.0)

			# divide every pixel value to be between 0 and 1
			for j in range(len(pxl)):
				pxl[j] = float(pxl[j])
				pxl[j] = pxl[j]/255

			# store the values to be returned
			labels.append(lbl)
			pixels.append(pxl)

	return labels, pixels

# get the labels and pixels from the
# MNIST_train.csv
labels, pixels = setupTrainSet()
# shuffle the data using zip so that
# both labels and image pixels
# are shuffled in-sync
rand = list(zip(labels, pixels))
random.shuffle(rand)
labels, pixels = zip(*rand)
labels = list(labels)
pixels = list(pixels)

# convert the train data list to numpy array
labels = np.array(labels)
pixels = np.array(pixels)

# get the labels and pixels from the
# MNIST_test.csv
testLabels, testPixels = setupTestSet()

# convert the test data list to numpy array
testLabels = np.array(testLabels)
testPixels = np.array(testPixels)

def sigmoid(x):
	return 1/ (1 + np.exp(-x))

outputUnits = 10
inputLength = len(pixels[0])

# setting the initial weights to be global so that
# all netowrks have the same starting point, this will
# yeild more consistent results since randomized weights
# for different neural networks in the same experiment
# can cause inconsistency and illogical results
initInputHiddenWeights = np.random.rand(100, inputLength) * (0.05 - (-0.05)) - 0.05
initHiddenOutputWeights = np.random.rand(outputUnits, 100+1) * (0.05 - (-0.05)) - 0.05

# the neuralNetwork class that houses
# the neural network structure and
# specifications
class neuralNetwork:

	inputToHiddenWeights = []
	deltaInputToHiddenWeights = []

	hiddenToOutputWeights = []
	deltaHiddenToOutputWeights = []
	hiddenValues = []
	hiddenErrorRate = []

	outputValues = []
	outputErrorRate = []

	y = []
	t = []

	learningRate = 0.1
	momentum = 0.9

	def __init__(self, numHiddenUnits, numOutputUnits, lengthOfInput):

		# initialize the lists to be numpy arrays
		# and copy the randomized initial weights
		# of the input->hidden AND hidden->output
		# to be between (-0.05, 0.05)

		# self.inputToHiddenWeights = \
		# np.random.rand(numHiddenUnits, lengthOfInput) * (0.05 - (-0.05)) - 0.05
		self.inputToHiddenWeights = \
		np.zeros(shape=(numHiddenUnits, lengthOfInput))

		# for i in range(len(self.inputToHiddenWeights)):
		# 	for j in range(len(self.inputToHiddenWeights[0])):
		# 		self.inputToHiddenWeights[i][j] = initInputHiddenWeights[i][j]

		np.copyto(self.inputToHiddenWeights, initInputHiddenWeights[:numHiddenUnits])

		self.deltaInputToHiddenWeights = \
		np.zeros(shape=(numHiddenUnits, lengthOfInput))

		# self.hiddenToOutputWeights = \
		# np.random.rand(numOutputUnits, numHiddenUnits+1) * (0.05 - (-0.05)) - 0.05

		self.hiddenToOutputWeights = \
		np.zeros(shape=(numOutputUnits, numHiddenUnits+1))

		# for i in range(len(self.hiddenToOutputWeights)):
		# 	for j in range(len(self.hiddenToOutputWeights[0])):
		# 		self.hiddenToOutputWeights[i][j] = initHiddenOutputWeights[i][j]

		np.copyto(self.hiddenToOutputWeights, initHiddenOutputWeights[:, :numHiddenUnits+1])

		self.deltaHiddenToOutputWeights = \
		np.zeros(shape=(numOutputUnits, numHiddenUnits+1))

		# initialize h1...hn
		# initialize o1...0n
		# initialize hbias to be 1
		self.hiddenValues = np.zeros(shape=(numHiddenUnits+1))
		self.hiddenValues[0] = 1
		self.hiddenErrorRate = np.zeros(shape=(numHiddenUnits))

		self.outputValues = np.zeros(shape=(numOutputUnits))
		self.outputErrorRate = np.zeros(shape=(numOutputUnits))

		self.y = []
		self.t = []
		self.learningRate = 0.1
		self.momentum = 0.9

	# the forward propagation function populates the
	# hidden and output values of its respective
	# neural network
	def forwardPropagation(self, pixels):

		for i in range (len(self.inputToHiddenWeights)):
			self.hiddenValues[i+1] = sigmoid(self.inputToHiddenWeights[i] @ pixels)

		for i in range(len(self.outputValues)):
			self.outputValues[i] = sigmoid(self.hiddenToOutputWeights[i] @ self.hiddenValues)

	# the back propagation function populates the
	# hidden and output error rates and the delte
	# weights input->hidden, hidden output
	# of its respective neural network
	def backPropagation(self, pixels, label):

		for i in range(len(self.outputErrorRate)):
			if label == i:
				self.outputErrorRate[i] = self.outputValues[i] * (1 - self.outputValues[i]) \
				* (0.9 - self.outputValues[i])
			else:
				self.outputErrorRate[i] = self.outputValues[i] * (1 - self.outputValues[i]) \
				* (0.1 - self.outputValues[i])

		for i in range(len(self.hiddenErrorRate)):
			self.hiddenErrorRate[i] = self.hiddenValues[i+1] * (1 - self.hiddenValues[i+1]) \
			* (self.hiddenToOutputWeights[:,i+1] @ self.outputErrorRate)

		for i in range(len(self.deltaHiddenToOutputWeights)):
			constans = self.learningRate * self.outputErrorRate[i]
			# for j in range(len(self.deltaHiddenToOutputWeights[i])):
			# 	self.deltaHiddenToOutputWeights[i][j] = constans * self.hiddenValues[j] \
			# 	+ self.momentum * self.deltaHiddenToOutputWeights[i][j]
			# 	self.hiddenToOutputWeights[i][j] += self.deltaHiddenToOutputWeights[i][j]
			self.deltaHiddenToOutputWeights[i] = \
			constans * self.hiddenValues + self.momentum * self.deltaHiddenToOutputWeights[i]
			self.hiddenToOutputWeights[i] \
			= self.hiddenToOutputWeights[i] + self.deltaHiddenToOutputWeights[i]

		for i in range(len(self.deltaInputToHiddenWeights)):
			constans = self.learningRate * self.hiddenErrorRate[i]
			# for j in range(len(self.deltaInputToHiddenWeights[i])):
			# 	self.deltaInputToHiddenWeights[i][j] = constans * pixels[j] \
			# 	+ self.momentum * self.deltaInputToHiddenWeights[i][j]
			# 	self.inputToHiddenWeights[i][j] += self.deltaInputToHiddenWeights[i][j]

			self.deltaInputToHiddenWeights[i] = \
			constans * pixels + self.momentum * self.deltaInputToHiddenWeights[i]
			self.inputToHiddenWeights[i] = \
			self.inputToHiddenWeights[i] + self.deltaInputToHiddenWeights[i]

	# the prediction result function returns
	# the neural network's prediction given
	# the input
	def predictionResult(self, pixels):

		HV = np.zeros(shape=(len(self.hiddenValues)))
		HV[0] = 1
		OV = {}

		for i in range (len(self.inputToHiddenWeights)):
			HV[i+1] = sigmoid(self.inputToHiddenWeights[i] @ pixels)

		for i in range(len(self.outputValues)):
			OV[i] = sigmoid(self.hiddenToOutputWeights[i] @ HV)

		return max(OV, key=OV.get)

# an auxiliary function that writes
# the accuracy data to an external CSV file
def writeAccuraciesToCSV(numHiddenUnits, percentageTrain, percentageTest):

	filename = str(numHiddenUnits)+'_hidden_units_accuracy.csv'
	with open(filename, 'w') as write:
		write = csv.writer(write)
		write.writerow(['epoch', '% train set', '% test set'])
		for i in range(len(percentageTrain)):
			write.writerow([i, percentageTrain[i], percentageTest[i]])

# an auxiliary function that writes the
# confusion matrix data to an external CSV file
def writeConfusionMatrixToCSV(numHiddenUnits, confusionMatrix):
	filename = str(numHiddenUnits)+'_hidden_units_confusion_matrix.csv'
	with open(filename, 'w') as write:
		write = csv.writer(write)
		write.writerows(confusionMatrix)

# experiment one function that sets up
# three different neural networks with a
# fixed learning rate (0.1)
# and momentum (0.9), but a different
# hidden unit length for every neural network

# neural network 1: 20 hidden units
# neural network 2: 50 hidden units
# neural network 3: 100 hidden units
def experimentOne():

	# setup the initial variables needed for the
	# networks to run

	outputUnits = 10
	inputLength = len(pixels[0])

	twentyUnitsNetwork = neuralNetwork(20, outputUnits, inputLength)
	twentyUnitsTrainPercentages = []
	twentyUnitsTestPercentages = []
	twentyUnitsConfusionMatrix = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

	fiftyUnitsNetwork = neuralNetwork(50, outputUnits, inputLength)
	fiftyUnitsTrainPercentages = []
	fiftyUnitsTestPercentages = []
	fiftyUnitsConfusionMatrix = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

	hundredUnitsNetwork = neuralNetwork(100, outputUnits, inputLength)
	hundredUnitsTrainPercentages = []
	hundredUnitsTestPercentages = []
	hundredUnitsConfusionMatrix = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

	# train the three networks for 51 epochs
	for epoch in range(51):

		print('========================== Experiment One ==========================\n')
		print('epoch #{}'.format(epoch))
		for i in range(len(pixels)):

			if i % 10000 == 0: print(i)

			twentyUnitsNetwork.forwardPropagation(pixels[i])
			twentyUnitsNetwork.backPropagation(pixels[i], labels[i])

			fiftyUnitsNetwork.forwardPropagation(pixels[i])
			fiftyUnitsNetwork.backPropagation(pixels[i], labels[i])

			hundredUnitsNetwork.forwardPropagation(pixels[i])
			hundredUnitsNetwork.backPropagation(pixels[i], labels[i])

		# setup the necessary values to compute accuracy
		twentyCorrectTrain = 0
		twentyCorrectTest = 0

		fiftyCorrectTrain= 0
		fiftyCorrectTest = 0

		hundredCorrectTrain= 0
		hundredCorrectTest = 0

		totalTrain = len(pixels)
		totalTest = len(testPixels)

		# compute the accuracy on the training set
		for i in range(len(pixels)):

			twentyCorrectTrain += 1 if \
			twentyUnitsNetwork.predictionResult(pixels[i]) == labels[i]\
			else 0
			fiftyCorrectTrain += 1 if \
			fiftyUnitsNetwork.predictionResult(pixels[i]) == labels[i]\
			else 0
			hundredCorrectTrain += 1 if \
			hundredUnitsNetwork.predictionResult(pixels[i]) == labels[i]\
			else 0

		# compute the accuracy on the testing set
		for i in range(len(testPixels)):

			twentyTestPrediction = \
			twentyUnitsNetwork.predictionResult(testPixels[i])
			fiftyTestPrediction = \
			fiftyUnitsNetwork.predictionResult(testPixels[i])
			hundredTestPrediction = \
			hundredUnitsNetwork.predictionResult(testPixels[i])

			twentyCorrectTest += 1 if twentyTestPrediction == testLabels[i] else 0
			fiftyCorrectTest += 1 if fiftyTestPrediction == testLabels[i] else 0
			hundredCorrectTest += 1 if hundredTestPrediction == testLabels[i] else 0

			# compute the confusion matrix on epoch 50
			if epoch == 50:
				twentyUnitsConfusionMatrix[testLabels[i]][twentyTestPrediction]+=1
				fiftyUnitsConfusionMatrix[testLabels[i]][fiftyTestPrediction]+=1
				hundredUnitsConfusionMatrix[testLabels[i]][hundredTestPrediction]+=1

		# store the computed accuracy in the accuracies list for graphing
		twentyUnitsPercentageTrain = round((twentyCorrectTrain/totalTrain)*100, 2)
		twentyUnitsPercentageTest = round((twentyCorrectTest/totalTest)*100, 2)
		twentyUnitsTrainPercentages.append(twentyUnitsPercentageTrain)
		twentyUnitsTestPercentages.append(twentyUnitsPercentageTest)

		fiftyUnitsPercentageTrain = round((fiftyCorrectTrain/totalTrain)*100, 2)
		fiftyUnitsPercentageTest = round((fiftyCorrectTest/totalTest)*100, 2)
		fiftyUnitsTrainPercentages.append(fiftyUnitsPercentageTrain)
		fiftyUnitsTestPercentages.append(fiftyUnitsPercentageTest)

		hundredUnitsPercentageTrain = round((hundredCorrectTrain/totalTrain)*100, 2)
		hundredUnitsPercentageTest = round((hundredCorrectTest/totalTest)*100, 2)
		hundredUnitsTrainPercentages.append(hundredUnitsPercentageTrain)
		hundredUnitsTestPercentages.append(hundredUnitsPercentageTest)

		# display the accuracy results of the epoch to std output
		print('Results for 20: train%: {}, test%: {}'.format(\
			twentyUnitsPercentageTrain, twentyUnitsPercentageTest))
		print('Results for 50: train%: {}, test%: {}'.format(\
			fiftyUnitsPercentageTrain, fiftyUnitsPercentageTest))
		print('Results for 100: train%: {}, test%: {}'.format(\
			hundredUnitsPercentageTrain, hundredUnitsPercentageTest))

	# write the accuracies to an external CSV file
	writeAccuraciesToCSV(20, twentyUnitsTrainPercentages, twentyUnitsTestPercentages)
	writeAccuraciesToCSV(50, fiftyUnitsTrainPercentages, fiftyUnitsTestPercentages)
	writeAccuraciesToCSV(100, hundredUnitsTrainPercentages, hundredUnitsTestPercentages)

	# write the confusion matrices to an external CSV file
	writeConfusionMatrixToCSV(20, twentyUnitsConfusionMatrix)
	writeConfusionMatrixToCSV(50, fiftyUnitsConfusionMatrix)
	writeConfusionMatrixToCSV(100, hundredUnitsConfusionMatrix)

	# print the confusion matrices to std output
	print('Confusion matrix results - 20 hidden units')
	print(pd.DataFrame(twentyUnitsConfusionMatrix))
	print()
	print('Confusion matrix results - 50 hidden units')
	print(pd.DataFrame(fiftyUnitsConfusionMatrix))
	print()
	print('Confusion matrix results - 100 hidden units')
	print(pd.DataFrame(hundredUnitsConfusionMatrix))

# an auxiliary function that stores
# the indices for half/quarter
# the training datasets with respect
# to the sample size being distributed
# equally across the 10 digits
def getHalfANDQuarterDataSet():
	# half the dataset section

	# testEquality dictionary ensures that
	# all digits are picked equally
	testEquality = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0\
	, 6: 0, 7: 0, 8: 0, 9: 0}
	halfDataIndexes = []
	# get the number splits the data equally in half
	# 60000/10 => /2 = 3000 sample per digit
	halfData = (len(pixels)/10)/2

	for i in range(len(pixels)):
		# if the digit has over 3000 samples, then don't store its index
		if labels[i] in [num for num, occurrences in testEquality.items() if occurrences >= halfData]:
			continue
		testEquality[labels[i]] += 1
		halfDataIndexes.append(i)
	# print(testEquality)

#=====================================================
	# quarter the dataset section

	# testEquality dictionary ensures that
	# all digits are picked equally
	testEquality = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0\
	, 6: 0, 7: 0, 8: 0, 9: 0}
	quarterDataIndexes = []
	# get the number splits the data equally in half
	# 60000/10 => /2 = 1500 sample per digit
	quarterData = (len(pixels)/10)/4

	for i in range(len(pixels)):
		# if the digit has over 1500 samples, then don't store its index
		if labels[i] in [num for num, occurrences in testEquality.items() if occurrences >= quarterData]:
			continue
		testEquality[labels[i]] += 1
		quarterDataIndexes.append(i)
	# print(testEquality)

	return quarterDataIndexes, halfDataIndexes

# experiment function function that sets up
# two different neural networks with a
# fixed learning rate (0.1)
# and momentum (0.9), but a different
# training dataset length
# for every neural network

# neural network 1: 1/4 training samples
# neural network 2: 1/2 training samples
def experimentTwo():

	# setup the initial variables needed for the
	# networks to run

	quarterDataIndexes, halfDataIndexes = getHalfANDQuarterDataSet()
	# print(len(quarterDataIndexes))
	# print(len(halfDataIndexes))

	outputUnits = 10
	inputLength = len(pixels[0])

	quarterDataHundredUnitsNetwork = neuralNetwork(100, outputUnits, inputLength)
	quarterDataTrainPercentages = []
	quarterDataTestPercentages = []
	quarterDataConfusionMatrix = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

	halfDataHundredUnitsNetwork = neuralNetwork(100, outputUnits, inputLength)
	halfDataTrainPercentages = []
	halfDataTestPercentages = []
	halfDataConfusionMatrix = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

	# train the three networks for 51 epochs
	for epoch in range(51):

		print('========================== Experiment Two ==========================\n')
		print('epoch #{}'.format(epoch))
		# testEquality = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0\
		# , 6: 0, 7: 0, 8: 0, 9: 0}

		for i in range(len(quarterDataIndexes)):

			quarterDataHundredUnitsNetwork.forwardPropagation(pixels[quarterDataIndexes[i]])
			quarterDataHundredUnitsNetwork.backPropagation(pixels[quarterDataIndexes[i]]\
				, labels[quarterDataIndexes[i]])
			# testEquality[labels[quarterDataIndexes[i]]] += 1

		# print(testEquality)
		# testEquality = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0\
		# , 6: 0, 7: 0, 8: 0, 9: 0}

		for i in range(len(halfDataIndexes)):

			halfDataHundredUnitsNetwork.forwardPropagation(pixels[halfDataIndexes[i]])
			halfDataHundredUnitsNetwork.backPropagation(pixels[halfDataIndexes[i]]\
				, labels[halfDataIndexes[i]])
			# testEquality[labels[halfDataIndexes[i]]] += 1

		# setup the necessary values to compute accuracy
		quarterCorrectTrain = 0
		quarterCorrectTest = 0

		halfCorrectTrain = 0
		halfCorrectTest = 0

		totalQuarter = len(quarterDataIndexes)
		totalHalf = len(halfDataIndexes)
		totalTest = len(testPixels)

		# compute the accuracy on 1/4 the training set
		for i in range(len(quarterDataIndexes)):

			quarterCorrectTrain += 1 if \
			quarterDataHundredUnitsNetwork.predictionResult(pixels[quarterDataIndexes[i]]) == \
			labels[quarterDataIndexes[i]] else 0

		# compute the accuracy on 1/2 the training set
		for i in range(len(halfDataIndexes)):

			halfCorrectTrain += 1 if \
			halfDataHundredUnitsNetwork.predictionResult(pixels[halfDataIndexes[i]]) == \
			labels[halfDataIndexes[i]] else 0

		for i in range(len(testPixels)):

			quarterTestPrediction = \
			quarterDataHundredUnitsNetwork.predictionResult(testPixels[i])

			halfTestPrediction = \
			halfDataHundredUnitsNetwork.predictionResult(testPixels[i])

			quarterCorrectTest += 1 if quarterTestPrediction == testLabels[i] else 0
			halfCorrectTest += 1 if halfTestPrediction == testLabels[i] else 0

			# compute the accuracy on the training set
			if epoch == 50:
				quarterDataConfusionMatrix[testLabels[i]][quarterTestPrediction]+=1
				halfDataConfusionMatrix[testLabels[i]][halfTestPrediction]+=1

		# store the computed accuracy in the accuracies list for graphing
		quarterDataPercentageTrain = round((quarterCorrectTrain/totalQuarter)*100, 2)
		quarterDataPercentageTest = round((quarterCorrectTest/totalTest)*100, 2)
		quarterDataTrainPercentages.append(quarterDataPercentageTrain)
		quarterDataTestPercentages.append(quarterDataPercentageTest)

		halfDataPercentageTrain = round((halfCorrectTrain/totalHalf)*100, 2)
		halfDataPercentageTest = round((halfCorrectTest/totalTest)*100, 2)
		halfDataTrainPercentages.append(halfDataPercentageTrain)
		halfDataTestPercentages.append(halfDataPercentageTest)

		# display the accuracy results of the epoch to std output
		print('Results for 1/4 data: train%: {}, test%: {}'.format(\
			quarterDataPercentageTrain, quarterDataPercentageTest))
		print('Results for 1/2 data: train%: {}, test%: {}'.format(\
			halfDataPercentageTrain, halfDataPercentageTest))

	# write the accuracies to an external CSV file
	writeAccuraciesToCSV('quarter_data_100', quarterDataTrainPercentages, quarterDataTestPercentages)
	writeAccuraciesToCSV('half_data_100', halfDataTrainPercentages, halfDataTestPercentages)

	# write the confusion matrices to an external CSV file
	writeConfusionMatrixToCSV('quarter_data_100', quarterDataConfusionMatrix)
	writeConfusionMatrixToCSV('half_data_100', halfDataConfusionMatrix)

	# print the confusion matrices to std output
	print('Confusion matrix results - 1/4 training data')
	print(pd.DataFrame(quarterDataConfusionMatrix))
	print()
	print('Confusion matrix results - 1/2 training data')
	print(pd.DataFrame(halfDataConfusionMatrix))

# main calls both of the experiments sequentially
def main():

	experimentOne()
	experimentTwo()

main()

# print total execution time
print("--- %s seconds ---" % (time.time() - start_time))
