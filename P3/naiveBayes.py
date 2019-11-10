import sys
import numpy as np

# the naiveBayes class houses
# the naive bayes structure and
# specifications
class naiveBayes:

	numClasses = None
	uniqueClasses = None

	# when constructing the class,
	# assign the number of unique classes
	# that the training set has
	def __init__(self, numClasses, classes):
		self.numClasses = numClasses
		self.uniqueClasses = {}
		for i in classes:
			self.uniqueClasses[int(i)] = {}

