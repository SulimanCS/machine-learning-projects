import sys
import os
import csv
import time
import numpy
import random
import pandas as pd
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
# using zip, both lists (labels and image pixels)
# are shuffled in sync so that labels won't
# be linked to the wrong image
rand = list(zip(labels, pixels))
random.shuffle(rand)
labels, pixels = zip(*rand)
labels = list(labels)
pixels = list(pixels)

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

def multBy100(listt):

	for i in range(len(listt)):
		listt[i] = listt[i] * 100
	return listt

def writeToCSV(learningRateRound, percentageTrain, percentageTest):

	filename = 'learning_round_'+str(learningRateRound)+'_accuracy.csv'
	with open(filename, 'w') as write:
		write = csv.writer(write)
		write.writerow(['epoch', '% train set', '% test set'])
		for i in range(len(percentageTrain)):
			write.writerow([i, percentageTrain[i], percentageTest[i]])

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
		confusionMatrix = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
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

		for epoch in range(51):
			print('====================================================\n')
			print('Learning round #{}, epoch #{}'.format(learningRateRound, epoch))
			for i in range(len(pixels)):
				p0.compute(labels[i], pixels[i], i, learningRate)
				p1.compute(labels[i], pixels[i], i, learningRate)
				p2.compute(labels[i], pixels[i], i, learningRate)
				p3.compute(labels[i], pixels[i], i, learningRate)
				p4.compute(labels[i], pixels[i], i, learningRate)
				p5.compute(labels[i], pixels[i], i, learningRate)
				p6.compute(labels[i], pixels[i], i, learningRate)
				p7.compute(labels[i], pixels[i], i, learningRate)
				p8.compute(labels[i], pixels[i], i, learningRate)
				p9.compute(labels[i], pixels[i], i, learningRate)

			correct = 0
			total = 0

			for i in range(len(pixels)):

				largest = {}

				res0 = p0.calc(pixels[i], i)
				largest[0] = res0

				res1 = p1.calc(pixels[i], i)
				largest[1] = res1

				res2 = p2.calc(pixels[i], i)
				largest[2] = res2

				res3 = p3.calc(pixels[i], i)
				largest[3] = res3

				res4 = p4.calc(pixels[i], i)
				largest[4] = res4

				res5 = p5.calc(pixels[i], i)
				largest[5] = res5

				res6 = p6.calc(pixels[i], i)
				largest[6] = res6

				res7 = p7.calc(pixels[i], i)
				largest[7] = res7

				res8 = p8.calc(pixels[i], i)
				largest[8] = res8

				res9 = p9.calc(pixels[i], i)
				largest[9] = res9

				if labels[i] == max(largest, key=largest.get):
					#print('correct output')
					correct += 1
				else:
					#print('wrong output')
					correct += 0

				total += 1

			accuracyTrain = round(correct/total, 3)

			correct = 0
			total = 0

			for i in range(len(testPixels)):

				largest = {}

				res0 = p0.calc(testPixels[i], i)
				largest[0] = res0

				res1 = p1.calc(testPixels[i], i)
				largest[1] = res1

				res2 = p2.calc(testPixels[i], i)
				largest[2] = res2

				res3 = p3.calc(testPixels[i], i)
				largest[3] = res3

				res4 = p4.calc(testPixels[i], i)
				largest[4] = res4

				res5 = p5.calc(testPixels[i], i)
				largest[5] = res5

				res6 = p6.calc(testPixels[i], i)
				largest[6] = res6

				res7 = p7.calc(testPixels[i], i)
				largest[7] = res7

				res8 = p8.calc(testPixels[i], i)
				largest[8] = res8

				res9 = p9.calc(testPixels[i], i)
				largest[9] = res9

				if testLabels[i] == max(largest, key=largest.get):
					#print('correct output')
					correct += 1
				else:
					#print('wrong output')
					correct += 0

				confusionMatrix[testLabels[i]][max(largest, key=largest.get)]+=1
				total += 1

			accuracyTest = round(correct/total, 3)

			print('Confusion matrix for learning round #{}'.format(learningRateRound))
			print(pd.DataFrame(confusionMatrix))
			with open('learning_round_'+str(learningRateRound)+'_confusion_matrix.txt', 'w') as f:
				print(pd.DataFrame(confusionMatrix), file=f)

			percentageTrain.append(accuracyTrain)
			percentageTest.append(accuracyTest)
			print('train set accuracy is: {}'.format(accuracyTrain))
			print('test set accuracy is: {}'.format(accuracyTest))
			print('\n')

		percentageTrain = multBy100(percentageTrain)
		percentageTest = multBy100(percentageTest)
		writeToCSV(learningRateRound, percentageTrain, percentageTest)

main()
print("--- %s seconds ---" % (time.time() - start_time))
