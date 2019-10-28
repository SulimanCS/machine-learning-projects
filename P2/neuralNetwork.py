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
		# and randomize the initial weights
		# of the input->hidden AND hidden->output
		# to be between (-0.05, 0.05)
		self.inputToHiddenWeights = \
		np.random.rand(numHiddenUnits, lengthOfInput) * (0.05 - (-0.05)) - 0.05
		self.deltaInputToHiddenWeights = \
		np.zeros(shape=(numHiddenUnits, lengthOfInput))

		self.hiddenToOutputWeights = \
		np.random.rand(numOutputUnits, numHiddenUnits+1) * (0.05 - (-0.05)) - 0.05
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

		# # REMOVE, JUST FOR TESTING
		# self.inputToHiddenWeights = np.zeros(shape=(numHiddenUnits, lengthOfInput))
		# self.hiddenToOutputWeights = np.zeros(shape=(numOutputUnits, numHiddenUnits+1))
		# self.inputToHiddenWeights[0][0] = -0.4
		# self.inputToHiddenWeights[0][1] = 0.2
		# self.inputToHiddenWeights[0][2] = 0.1

		# self.inputToHiddenWeights[1][0] = -0.2
		# self.inputToHiddenWeights[1][1] = 0.4
		# self.inputToHiddenWeights[1][2] = -0.1

		# self.hiddenToOutputWeights[0][0] = 0.1
		# self.hiddenToOutputWeights[0][1] = -0.2
		# self.hiddenToOutputWeights[0][2] = 0.1

		# self.hiddenToOutputWeights[1][0] = 0.4
		# self.hiddenToOutputWeights[1][1] = -0.1
		# self.hiddenToOutputWeights[1][2] = 0.1

	def forwardPropagation(self, pixels):

		for i in range (len(self.inputToHiddenWeights)):
			self.hiddenValues[i+1] = sigmoid(self.inputToHiddenWeights[i] @ pixels)

		for i in range(len(self.outputValues)):
			self.outputValues[i] = sigmoid(self.hiddenToOutputWeights[i] @ self.hiddenValues)

	def backPropagation(self, pixels, label):

		# print(self.hiddenValues)
		# print(self.outputValues)
		for i in range(len(self.outputErrorRate)):
			if label == i:
				self.outputErrorRate[i] = self.outputValues[i] * (1 - self.outputValues[i]) \
				* (0.9 - self.outputValues[i])
			else:
				self.outputErrorRate[i] = self.outputValues[i] * (1 - self.outputValues[i]) \
				* (0.1 - self.outputValues[i])

		# print(self.outputErrorRate)

		for i in range(len(self.hiddenErrorRate)):
			self.hiddenErrorRate[i] = self.hiddenValues[i+1] * (1 - self.hiddenValues[i+1]) \
			* (self.hiddenToOutputWeights[:,i+1] @ self.outputErrorRate)
			# print(self.hiddenToOutputWeights[:,i+1], self.outputErrorRate)
			# print(self.hiddenToOutputWeights[:,i+1] @ self.outputErrorRate)
		# print(self.hiddenErrorRate)

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

		# print(self.deltaHiddenToOutputWeights)
		# print(self.hiddenToOutputWeights)
		# print(self.hiddenValues)
		# print(len(self.deltaInputToHiddenWeights))
		# print(len(self.deltaInputToHiddenWeights[0]))
		for i in range(len(self.deltaInputToHiddenWeights)):
			# print('i is: {}'.format(i))
			constans = self.learningRate * self.hiddenErrorRate[i]
			# print(self.hiddenErrorRate)
			# for j in range(len(self.deltaInputToHiddenWeights[i])):
			# 	self.deltaInputToHiddenWeights[i][j] = constans * pixels[j] \
			# 	+ self.momentum * self.deltaInputToHiddenWeights[i][j]
			# 	self.inputToHiddenWeights[i][j] += self.deltaInputToHiddenWeights[i][j]
			# print(type(self.deltaInputToHiddenWeights))
			# print(type(constans))
			# print(type(pixels))
			# print(pixels)
			# print(type(pixels[0]))
			# print(type(self.momentum))
			# exit(1)
			self.deltaInputToHiddenWeights[i] = \
			constans * pixels + self.momentum * self.deltaInputToHiddenWeights[i]
			self.inputToHiddenWeights[i] = \
			self.inputToHiddenWeights[i] + self.deltaInputToHiddenWeights[i]

	def predictionResult(self, pixels):

		HV = np.zeros(shape=(len(self.hiddenValues)))
		HV[0] = 1
		OV = {}
		# print(len(HV))
		# print(len(OV))
		for i in range (len(self.inputToHiddenWeights)):
			HV[i+1] = sigmoid(self.inputToHiddenWeights[i] @ pixels)

		for i in range(len(self.outputValues)):
			OV[i] = sigmoid(self.hiddenToOutputWeights[i] @ HV)
		# return OV
		# return label == max(OV, key=OV.get)
		return max(OV, key=OV.get)

def writeAccuraciesToCSV(numHiddenUnits, percentageTrain, percentageTest):

	filename = str(numHiddenUnits)+'_hidden_units_accuracy.csv'
	with open(filename, 'w') as write:
		write = csv.writer(write)
		write.writerow(['epoch', '% train set', '% test set'])
		for i in range(len(percentageTrain)):
			write.writerow([i, percentageTrain[i], percentageTest[i]])

def writeConfusionMatrixToCSV(numHiddenUnits, confusionMatrix):
	filename = str(numHiddenUnits)+'_hidden_units_confusion_matrix.csv'
	with open(filename, 'w') as write:
		write = csv.writer(write)
		write.writerows(confusionMatrix)

def experimentOne():

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

		twentyCorrectTrain = 0
		twentyCorrectTest = 0

		fiftyCorrectTrain= 0
		fiftyCorrectTest = 0

		hundredCorrectTrain= 0
		hundredCorrectTest = 0

		totalTrain = len(pixels)
		totalTest = len(testPixels)

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

			if epoch == 50:
				twentyUnitsConfusionMatrix[testLabels[i]][twentyTestPrediction]+=1
				fiftyUnitsConfusionMatrix[testLabels[i]][fiftyTestPrediction]+=1
				hundredUnitsConfusionMatrix[testLabels[i]][hundredTestPrediction]+=1

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

		print('Results for 20: train%: {}, test%: {}'.format(\
			twentyUnitsPercentageTrain, twentyUnitsPercentageTest))
		print('Results for 50: train%: {}, test%: {}'.format(\
			fiftyUnitsPercentageTrain, fiftyUnitsPercentageTest))
		print('Results for 100: train%: {}, test%: {}'.format(\
			hundredUnitsPercentageTrain, hundredUnitsPercentageTest))

	writeAccuraciesToCSV(20, twentyUnitsTrainPercentages, twentyUnitsTestPercentages)
	writeAccuraciesToCSV(50, fiftyUnitsTrainPercentages, fiftyUnitsTestPercentages)
	writeAccuraciesToCSV(100, hundredUnitsTrainPercentages, hundredUnitsTestPercentages)

	writeConfusionMatrixToCSV(20, twentyUnitsConfusionMatrix)
	writeConfusionMatrixToCSV(50, fiftyUnitsConfusionMatrix)
	writeConfusionMatrixToCSV(100, hundredUnitsConfusionMatrix)

def getHalfANDQuarterDataSet():

	testEquality = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0\
	, 6: 0, 7: 0, 8: 0, 9: 0}
	halfDataIndexes = []
	halfData = (len(pixels)/10)/2

	for i in range(len(pixels)):
		if labels[i] in [num for num, occurrences in testEquality.items() if occurrences >= halfData]:
			continue
		testEquality[labels[i]] += 1
		halfDataIndexes.append(i)
	# print(testEquality)

#=====================================================

	testEquality = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0\
	, 6: 0, 7: 0, 8: 0, 9: 0}
	quarterDataIndexes = []
	quarterData = (len(pixels)/10)/4

	for i in range(len(pixels)):
		if labels[i] in [num for num, occurrences in testEquality.items() if occurrences >= quarterData]:
			continue
		testEquality[labels[i]] += 1
		quarterDataIndexes.append(i)
	# print(testEquality)

	return quarterDataIndexes, halfDataIndexes

def experimentTwo():


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


