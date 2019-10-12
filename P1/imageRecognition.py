import sys
import os
import csv
import time
import numpy
def setup():

	global TRAINSET
	TRAINSET = 'mnist_test.csv'
	#TRAINSET = 'mnist_train.csv'
	dirr = os.path.dirname(__file__)
	path = os.path.join(dirr, TRAINSET)

	# initialize the lables/output and pixels lists
	# the lables are meant to be 1D list, and the pixels 2D
	labels = []
	pixels = []

	# read in the lables and images info
	with open(path, 'r') as fil:
		r = csv.reader(fil)
		for line in r:
				labels.append(line[0])
				pixels.append(line[1:])

	print(len(labels))
	print(len(pixels))

	# preprocessing, go through every pixel of every image
	for i in range(len(pixels)):
		# insert the bias at the beginning of the array
		pixels[i].insert(0, 1.0)
		for j in range (len(pixels[i])):
			if j == 0:
				continue
			# divide every pixel value to be between 0 and 1
			pixels[i][j] = numpy.float16(pixels[i][j])
			pixels[i][j]= pixels[i][j]/255
			pixels[i][j] = numpy.float16(pixels[i][j])

	print(len(pixels))
	print(len(pixels[0]))

	return labels, pixels
