import sys
import os
import csv
import time
import numpy
import random
# record when the program has started
start_time = time.time()

def setupTrainSet():

	global TRAINSET
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

	global TRAINSET
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

labels, pixels = setupTrainSet()
testLabels, testPixels = setupTestSet()

class perceptron:

	weights = []
	y = []
	t = []
	nclass = -1

	def __init__(self, nclass):

		self.weights = []
		for i in range(785):
			z = random.uniform(-0.5, 0.5)
			self.weights.append(z)

		self.y = []
		self.t = []
		self.nclass = nclass

	def compute(self, label, pixelsPerImage, imageIndex, learningRate):

		if label == self.nclass:
			tt = 1
			self.t.append(1)
			# if imageIndex % 10000 == 0:
			# 	print('match perceptron: {} @ {}'.format(self.nclass, imageIndex))
		else:
			tt = 0
			self.t.append(0)

		result = 0.0
		for i in range(len(pixelsPerImage)):
			result += self.weights[i] * pixelsPerImage[i]

		if result > 0:
			yy = 1
			self.y.append(1)
		else:
			yy = 0
			self.y.append(0)

		if (tt != yy):
			for z in range(len(pixelsPerImage)):
					self.weights[z] += learningRate * (tt - yy) * pixelsPerImage[z]

	def calc(self, pixelsPerImage, imageIndex):

		result = 0.0
		for j in range(len(pixelsPerImage)):
			result += self.weights[j] * pixelsPerImage[j]
		return result

def main():

	learningRate = 0
	for learningRateRound in range(1, 4):
		if learningRateRound == 1:
			learningRate = 0.001
		elif learningRateRound == 2:
			learningRate = 0.01
		else:
			learningRate = 0.1

		percentageTrain = []
		percentageTest = []
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

