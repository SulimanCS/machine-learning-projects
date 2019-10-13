import sys
import os
import csv
import time
import numpy
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
class perceptron:

	weights = []
	y = []
	t = []
	nclass = -1

	def __init__(self, nclass):

		self.weights = []
		for i in range(785):
			z = round(numpy.random.uniform(-0.5, 0.5), 4)
			self.weights.append(z)

		self.y = []
		self.t = []
		self.nclass = nclass

