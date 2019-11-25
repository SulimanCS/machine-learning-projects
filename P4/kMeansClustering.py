import numpy as np
import sys
import copy

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

		# out of the N runs, store the necessary data for the
		# best one
		bestRun = None

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

				# loop through every entry in the training set
				# save the row number in a different variable
				for rowNum, row in enumerate(rows):
					clusterResults = {key: [] for key in range(len(samples))}
					# print(row, '---', clusterResults)
					for sampleNum, sample in enumerate(samples):
						# print(sampleNum)
						# print(sample)
						res = 0
						for index in range(len(row)):
							res += np.square(np.sqrt(np.square(row[index] - sample[index])))
						clusterResults[sampleNum] = res
					minDistance = min(clusterResults, key=clusterResults.get)
					clusterMembership[minDistance].append(rowNum)

				# update cluster centers
				for clusterCenter in clusterMembership:
					if clusterMembership[clusterCenter] == []:
						samples[clusterCenter] = np.zeros(len(samples[0]), dtype=np.float64)
						# TODO change this after trying k=30
						print('yes')
						exit(1)
					else:
						clusterRows = rows[clusterMembership[clusterCenter]]
						for i in range(len(rows[0])):
							samples[clusterCenter, i] = np.sum(clusterRows[:, i]) / len(clusterMembership[clusterCenter])

				if oldClusterMembership == clusterMembership:
					AMSE = self.averageMeanSquareError(rows, samples, clusterMembership)
					if bestRun == None:
						bestRun = {'AMSE': AMSE, 'clusterCenters': copy.deepcopy(samples), \
						'clusterMembership': copy.deepcopy(clusterMembership)}
					else:
						if bestRun['AMSE'] > AMSE:
							print('YES')
							bestRun = {'AMSE': AMSE, 'clusterCenters': copy.deepcopy(samples), \
							'clusterMembership': copy.deepcopy(clusterMembership)}
							# exit(1)

					print('exiting, ', runNum)
					# exit(1)
					break
				if oldClusterMembership == None or oldClusterMembership != clusterMembership:
					# print('we are here')
					oldClusterMembership = copy.deepcopy(clusterMembership)

					# exit(1)
				# print('final samples:', samples)
				# print('final cl:', clusterMembership)
				tryy+=1
				# if tryy == 2:
				# 	exit(1)

	def averageMeanSquareError(self, data, clusterCenters, clusterMembership):

		# print('in averageMeanSquareError')
		averageMSE = []
		for i in clusterMembership:
			res=0
			for entry in clusterMembership[i]:
				finalEuclideanDistance=0
				for attributeNum in range(len(data[entry])):
					euclideanDistance = np.square(np.sqrt(np.square(data[entry, attributeNum] - clusterCenters[i, attributeNum])))
					finalEuclideanDistance+=euclideanDistance
				res+=finalEuclideanDistance
				# print(finalEuclideanDistance)
			averageMSE.append(res/len(clusterMembership[i]))

		return sum(averageMSE)/len(clusterMembership)

	def meanSquareSeparation(self, clusterCenters):
		# print('MSS')
		denamonitor = (len(clusterCenters)*(len(clusterCenters)-1))/2
		# print('m8am', denamonitor)
		res=0
		for i in range(len(clusterCenters)):
			# print('pair i', i, 'with')
			for j in range(i+1, len(clusterCenters)):
				# print(j)
				for z in range(len(clusterCenters[0])):
					euclideanDistance = np.square(np.sqrt(np.square(clusterCenters[i, z] - clusterCenters[j, z])))
					res+=euclideanDistance
		res = res/denamonitor
		# print('final: ', res)
		return res

# loads the dataset file elements as floats into a 2D numpy array
def loadSet(filename):

	# for our experiment, all the datasets attributes
	# are separated by a comma
	return np.loadtxt(filename, delimiter=',', dtype=np.float64)

