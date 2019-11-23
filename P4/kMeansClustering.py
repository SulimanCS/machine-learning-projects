import numpy as np
import sys

# the kMeansClustering class houses
# the K-Means CLustering structure
# and specifications
class kMeansClustering:

	k = None

	# when constructing the class,
	# assign the number of clusters K
	def __init__(self, k): self.k = k

	# picks k random samples from the supplied training set
	# and returns their indices
	def pickRandom(self, trainSet):

		return np.random.randint(len(trainSet), size=self.k)

	def train(self, trainSet):

		# get k random samples indices
		indices = self.pickRandom(trainSet)

		# copy the actual sample rows from the training set
		samples = np.copy(trainSet[indices])

		# store every row excluding the last column (class column)
		samples = samples[:, :-1]

		# store the last column of every row (class column)
		samplesLabels = samples[:, -1]

		# store every row excluding the last column (class column)
		rows = trainSet[:, :-1]

		# store the last column of every row (class column)
		labels = trainSet[:, -1]

		numRuns = 5

# loads the dataset file elements as floats into a 2D numpy array
def loadSet(filename):

	# for our experiment, all the datasets attributes
	# are separated by a comma
	return np.loadtxt(filename, delimiter=',', dtype=np.float64)

