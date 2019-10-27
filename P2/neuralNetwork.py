import sys
import os
import csv
import time
import numpy as np
import math
import random
import pandas as pd

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
	return 1/ (1 + math.exp(-x))

# the neuralNetwork class that houses
# the neural network structure and
# specifications
class neuralNetwork:

	inputToHiddenWeights = []

	hiddenToOutputWeights = []
	hiddenValues = []
	errorRateHidden = []

	outputValues = []
	errorRateOutput = []

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
		self.hiddenToOutputWeights = \
		np.random.rand(numOutputUnits, numHiddenUnits+1) * (0.05 - (-0.05)) - 0.05

		# initialize h1...hn
		# initialize o1...0n
		# initialize hbias to be 1
		self.hiddenValues = np.zeros(shape=(numHiddenUnits+1))
		self.hiddenValues[0] = 1
		self.errorRateHidden = np.zeros(shape=(numHiddenUnits))

		self.outputValues = np.zeros(shape=(numOutputUnits))
		self.errorRateOutput = np.zeros(shape=(numOutputUnits))

		self.y = []
		self.t = []

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

		for i in range(len(self.inputToHiddenWeights)):
			self.outputValues[i] = sigmoid(self.hiddenToOutputWeights[i] @ self.hiddenValues)

