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

# loads the dataset file elements as floats into a 2D numpy array
def loadSet(filename):

	# for our experiment, all the datasets attributes
	# are separated by a whitespace
	return np.loadtxt(filename, delimiter=None, dtype=np.float64)

