# ==============================================
# Name: Suliman Alsarraf
# Assignment: Project 1
# Class: CS445 Machine Learning
# ==============================================

import sys
import os
import csv
import time
import numpy as np
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

# the perceptron class that houses
# the weights, y (weights * pixels),
# t (target) and an integer that
# represents which digit does the
# instance of the perceptron represents
class perceptron:

	weights = []
	y = []
	t = []
	nclass = -1

	# the class constructor
	def __init__(self, nclass):

		# randomize the initial weights to be
		# between (-0.05, 0.05)
		self.weights = np.zeros(shape=(785))
		for i in range(785):
			z = random.uniform(-0.05, 0.05)
			self.weights[i] = z

		self.y = []
		self.t = []
		# the class digit representation is supplied
		self.nclass = nclass

	# a function that computes the weights, y and t values after
	# every epoch
	def computeNewWeights(self, label, pixelsPerImage, imageIndex, learningRate):

		# start_time = time.time()
		if label == self.nclass:
			tt = 1
			self.t.append(1)
			# if imageIndex % 10000 == 0:
			# 	print('match perceptron: {} @ {}'.format(self.nclass, imageIndex))
		else:
			tt = 0
			self.t.append(0)

		result = 0.0
		# store the sum of the two arrays product into result
		result = self.weights @ pixelsPerImage

		if result > 0:
			yy = 1
			self.y.append(1)
		else:
			yy = 0
			self.y.append(0)

		if (tt != yy):
			# 𝑤𝑖 ⟵ 𝑤𝑖 + 𝜂(𝑡^𝑘 − 𝑦^𝑘 )𝑥𝑖^𝑘 ,
			self.weights = self.weights + learningRate * (tt - yy) * pixelsPerImage

		# if imageIndex == 1000:
		# 	print("--- %s seconds ---" % (time.time() - start_time))

	# a function that computes the prediction ⟵ weights * x (pixels per image)
	def prediction(self, pixelsPerImage, imageIndex):

		result = 0.0
		# store the sum of the two arrays product into result
		# then return it, this is the prediction of a given
		# perceptron
		result = self.weights @ pixelsPerImage
		return result

# a function to adjust the percentages to be readable
def multBy100(listt):

	for i in range(len(listt)):
		listt[i] = listt[i] * 100
	return listt

# a function to write the accuracies of
# every learning rate to an external file
def writeToCSV(learningRateRound, percentageTrain, percentageTest):

	filename = 'learning_round_'+str(learningRateRound)+'_accuracy.csv'
	with open(filename, 'w') as write:
		write = csv.writer(write)
		write.writerow(['epoch', '% train set', '% test set'])
		for i in range(len(percentageTrain)):
			write.writerow([i, percentageTrain[i], percentageTest[i]])

# the main function where the algoirthm takes place
def main():

	learningRate = 0
	for learningRateRound in range(1, 4):
		# based on the current round, adjust
		# the learning rate (n) value
		if learningRateRound == 1:
			learningRate = 0.001
		elif learningRateRound == 2:
			learningRate = 0.01
		else:
			learningRate = 0.1

		# initialize the percentages lists for
		# the training and testing sets
		percentageTrain = []
		percentageTest = []
		# initialize the confusion matrix per learning rate
		confusionMatrix = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
		# initialize a perceptron per digit (0-9)
		p0 = perceptron(0)
		p1 = perceptron(1)
		p2 = perceptron(2)
		p3 = perceptron(3)
		p4 = perceptron(4)
		p5 = perceptron(5)
		p6 = perceptron(6)
		p7 = perceptron(7)
		p8 = perceptron(8)
		p9 = perceptron(9)

		# loop through the 0...51 epochs
		for epoch in range(51):
			print('====================================================\n')
			print('Learning round #{}, epoch #{}'.format(learningRateRound, epoch))
			# for every image/image, train and update the weights
			for i in range(len(pixels)):
				p0.computeNewWeights(labels[i], pixels[i], i, learningRate)
				p1.computeNewWeights(labels[i], pixels[i], i, learningRate)
				p2.computeNewWeights(labels[i], pixels[i], i, learningRate)
				p3.computeNewWeights(labels[i], pixels[i], i, learningRate)
				p4.computeNewWeights(labels[i], pixels[i], i, learningRate)
				p5.computeNewWeights(labels[i], pixels[i], i, learningRate)
				p6.computeNewWeights(labels[i], pixels[i], i, learningRate)
				p7.computeNewWeights(labels[i], pixels[i], i, learningRate)
				p8.computeNewWeights(labels[i], pixels[i], i, learningRate)
				p9.computeNewWeights(labels[i], pixels[i], i, learningRate)

			# correct/total calculates the accuracy
			correct = 0
			total = 0

			# for every image, compute w*x
			# per perceptron
			# the perceptron with the largest w*k
			# value is the perceptron that
			# represents the prediction

			# this loop is for the training set
			# accuracy, and predections
			for i in range(len(pixels)):

				largest = {}

				res0 = p0.prediction(pixels[i], i)
				largest[0] = res0

				res1 = p1.prediction(pixels[i], i)
				largest[1] = res1

				res2 = p2.prediction(pixels[i], i)
				largest[2] = res2

				res3 = p3.prediction(pixels[i], i)
				largest[3] = res3

				res4 = p4.prediction(pixels[i], i)
				largest[4] = res4

				res5 = p5.prediction(pixels[i], i)
				largest[5] = res5

				res6 = p6.prediction(pixels[i], i)
				largest[6] = res6

				res7 = p7.prediction(pixels[i], i)
				largest[7] = res7

				res8 = p8.prediction(pixels[i], i)
				largest[8] = res8

				res9 = p9.prediction(pixels[i], i)
				largest[9] = res9

				# if the prediction matches the label
				# then it means that the prediction is correct
				if labels[i] == max(largest, key=largest.get):
					#print('correct output')
					correct += 1
				# otherwise, the prediction is not correct
				else:
					#print('wrong output')
					correct += 0

				total += 1

			# compute the accuracy for the training set
			accuracyTrain = round(correct/total, 3)

			correct = 0
			total = 0

			# this loop is for the testing set
			# accuracy, and predections
			for i in range(len(testPixels)):

				largest = {}

				res0 = p0.prediction(testPixels[i], i)
				largest[0] = res0

				res1 = p1.prediction(testPixels[i], i)
				largest[1] = res1

				res2 = p2.prediction(testPixels[i], i)
				largest[2] = res2

				res3 = p3.prediction(testPixels[i], i)
				largest[3] = res3

				res4 = p4.prediction(testPixels[i], i)
				largest[4] = res4

				res5 = p5.prediction(testPixels[i], i)
				largest[5] = res5

				res6 = p6.prediction(testPixels[i], i)
				largest[6] = res6

				res7 = p7.prediction(testPixels[i], i)
				largest[7] = res7

				res8 = p8.prediction(testPixels[i], i)
				largest[8] = res8

				res9 = p9.prediction(testPixels[i], i)
				largest[9] = res9

				# if the prediction matches the label
				# then it means that the prediction is correct
				if testLabels[i] == max(largest, key=largest.get):
					#print('correct output')
					correct += 1
				# otherwise, the prediction is not correct
				else:
					#print('wrong output')
					correct += 0

				# if the algorithm reached the final epoch
				# in the learning rate, then compute the confusion
				# matrix
				if epoch == 50:
					confusionMatrix[testLabels[i]][max(largest, key=largest.get)]+=1

				total += 1

			# compute the accuracy for the training set
			accuracyTest = round(correct/total, 3)

			print('Confusion matrix for learning round #{}'.format(learningRateRound))

			# if the algorithm reached the final epoch
			# in the learning rate, then print and write
			# the confusion matrix to an external file
			# and label the external file with the
			# number of the learning round
			if epoch == 50:
				print(pd.DataFrame(confusionMatrix))

				filename = 'learning_round_'+str(learningRateRound)+'_confusion_matrix.txt'
				with open(filename, 'w') as f:
					print(pd.DataFrame(confusionMatrix), file=f)

			# store the accuracies of the training and testing sets
			percentageTrain.append(accuracyTrain)
			percentageTest.append(accuracyTest)

			print('train set accuracy is: {}'.format(accuracyTrain))
			print('test set accuracy is: {}'.format(accuracyTest))

			print('\n')

		# make the poercentages readable
		percentageTrain = multBy100(percentageTrain)
		percentageTest = multBy100(percentageTest)

		# write the accuracies to an external file
		writeToCSV(learningRateRound, percentageTrain, percentageTest)

		# print the accuracies to the standard output
		print(percentageTrain)
		print(percentageTest)


main()
print("--- %s seconds ---" % (time.time() - start_time))
