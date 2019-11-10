import sys
import numpy as np

# the naiveBayes class houses
# the naive bayes structure and
# specifications
class naiveBayes:

	numClasses = None
	numAttributes = None
	uniqueClasses = None

	# when constructing the class,
	# assign the number of unique classes
	# that the training set has
	def __init__(self, numClasses, numAttributes, classes):

		self.numClasses = numClasses
		self.numAttributes = numAttributes
		self.uniqueClasses = {}

		for i in classes:
			self.uniqueClasses[int(i)] = {}

	# create a probabilistic model
	def train(self, trainSet):

		# loop over all the classes in the supplied dataset
		for i in self.uniqueClasses.keys():

			# get location of rows that
			# only match the current class type
			seperator = np.squeeze(trainSet[:, -1]) == i

			# get the actual row values of the loop's class type
			row = trainSet[seperator, :]

			# remove the last column (since it is the class value column)
			row = row[:,:-1]

			# store the mean and standard deviation
			# of all columns
			mean = row.mean(axis=0)
			std = row.std(axis=0)

			# if there is any value in the std array
			# where it is less than 0.01, then
			# substitute that value with 0.01
			for j in np.where(std < 0.01)[0]:
				std[j] = 0.01

			# link the mean and std values
			# to their respective unique class
			self.uniqueClasses[i] = {
				'mean': mean,
				'std': std,
				'percentage': len(row)/len(trainSet)
			}

# loads the dataset file elements as floats into a 2D numpy array
def loadSet(filename):

	# for our experiment, all the datasets attributes
	# are separated by a whitespace
	return np.loadtxt(filename, delimiter=None, dtype=np.float64)

def main(trainSetFilename, testSetFilename):

	# load training & testing sets
	trainSet, testSet = loadSet(trainSetFilename), loadSet(testSetFilename)

	# get how many attributes there are in the given train set
	numAttributes = len(trainSet[0]) - 1

	# get the last column (class values) from the train set
	lastColumn = trainSet[:, len(trainSet[0])-1]

	# get only the unique values (eliminate duplicates)
	classes = set(lastColumn)

	# get how many classes there are for the given dataset
	# and create a naive bayes class with that number of classes,
	# classes, and number of attributes in the given set
	NB = naiveBayes(len(classes), numAttributes, classes)

