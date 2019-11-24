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

		# rows = [[0,1], [1,0], [2,0], [4,0]]
		# samples = [[1,1], [4,1]]
		# rows = np.asarray(rows, dtype=np.float64)
		# samples = np.asarray(samples, dtype=np.float64)

		# loop through the algorithm 5 times,
		# and pick the best run
		# the reason for this is because for every run
		# the chosen random samples are different
		for runNum in range(numRuns):
			# in every new run, create a new cluster membership

			# since the run starts at the first iteration
			# we don't have the results of the previous iteration
			# therefore, initialize to None
			oldClusterMembership = None

			tryy = 0
			# loop until the old cluster membership matches the current one
			while True:

				# for every iteration, initialize the cluster membership
				clusterMembership = {key: [] for key in range(len(samples))}

# loads the dataset file elements as floats into a 2D numpy array
def loadSet(filename):

	# for our experiment, all the datasets attributes
	# are separated by a comma
	return np.loadtxt(filename, delimiter=',', dtype=np.float64)

