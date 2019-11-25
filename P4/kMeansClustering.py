import numpy as np
import sys
import copy
import csv
import matplotlib.pyplot as plt
import pandas as pd

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

		# out of the N runs, store the necessary data for the
		# best one
		bestRun = None

		# loop through the algorithm 5 times,
		# and pick the best run
		# the reason for this is because for every run
		# the chosen random samples are different
		for runNum in range(numRuns):

			# in every run, pick different random cluster centers

			# get k random samples indices
			indices = self.pickRandom(trainSet)

			# copy the actual sample rows from the training set
			samples = np.copy(trainSet[indices])

			# store every row excluding the last column (class column)
			samples = samples[:, :-1]

			# store the last column of every row (class column)
			samplesLabels = samples[:, -1]


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
					for sampleNum, sample in enumerate(samples):
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
							bestRun = {'AMSE': AMSE, 'clusterCenters': copy.deepcopy(samples), \
							'clusterMembership': copy.deepcopy(clusterMembership)}

					print('exiting, ', runNum)
					break
				if oldClusterMembership == None or oldClusterMembership != clusterMembership:
					oldClusterMembership = copy.deepcopy(clusterMembership)

				tryy+=1
				# print('try:', tryy, '  run num:', runNum)

		bestRun['MSS'] = self.meanSquareSeparation(bestRun['clusterCenters'])
		bestRun['meanEntropy'] = self.meanEntropy(trainSet, bestRun['clusterMembership'])
		print('final values')
		print('ASME', bestRun['AMSE'])
		print('MSS', bestRun['MSS'])
		print('MEAN ENTROPY', bestRun['meanEntropy'])
		return bestRun

	def averageMeanSquareError(self, data, clusterCenters, clusterMembership):

		averageMSE = []
		for i in clusterMembership:
			res=0
			for entry in clusterMembership[i]:
				finalEuclideanDistance=0
				for attributeNum in range(len(data[entry])):
					euclideanDistance = np.square(np.sqrt(np.square(data[entry, attributeNum] - clusterCenters[i, attributeNum])))
					finalEuclideanDistance+=euclideanDistance
				res+=finalEuclideanDistance
			averageMSE.append(res/len(clusterMembership[i]))

		return sum(averageMSE)/len(clusterMembership)

	def meanSquareSeparation(self, clusterCenters):

		denamonitor = (len(clusterCenters)*(len(clusterCenters)-1))/2
		res=0
		for i in range(len(clusterCenters)):
			for j in range(i+1, len(clusterCenters)):
				for z in range(len(clusterCenters[0])):
					euclideanDistance = np.square(np.sqrt(np.square(clusterCenters[i, z] - clusterCenters[j, z])))
					res+=euclideanDistance
		res = res/denamonitor
		return res

	def meanEntropy(self, trainSet, clusterMembership):

		entropy = {key: 0 for key in clusterMembership}
		for cluster in clusterMembership:
			clusterDistribution = {}
			total=0
			for index in clusterMembership[cluster]:
				lastColumn = len(trainSet[index]) - 1
				if trainSet[index, lastColumn] not in clusterDistribution:
					clusterDistribution[trainSet[index, lastColumn]] = 1
				else:
					clusterDistribution[trainSet[index, lastColumn]]+=1

				total+=1
			res=0
			for i in clusterDistribution:
				if total != 0:
					res+=clusterDistribution[i]/total * np.log2(clusterDistribution[i]/total)
			res = res * -1
			entropy[cluster] = res

		meanEntropy = 0
		for cluster in clusterMembership:
			meanEntropy += (len(clusterMembership[cluster])/len(trainSet)) \
			* entropy[cluster]
		return meanEntropy

	def classify(self, trainSet, testSet, clusterMembership, clusterCenters):

		# associate each cluster center with the most frequent class it contains
		frequentClassPerCluster = {}
		lastIndex = len(trainSet[0]) - 1
		for cluster in clusterMembership:
			clusterLabels = {}
			for index in clusterMembership[cluster]:
				classs = trainSet[index, lastIndex]
				if classs not in clusterLabels:
					clusterLabels[classs]=1
				else:
					clusterLabels[classs]+=1

			frequentClassPerCluster[cluster] = int(max(clusterLabels, key=clusterLabels.get))

		# assign each test instance the class of the closest cluster center
		# compute accuracy and confusion matrix
		confusionMatrix = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

		correct=0
		total=len(testSet)

		# store every row excluding the last column (class column)
		rows = testSet[:, :-1]

		# store the last column of every row (class column)
		labels = testSet[:, -1]

		# loop through every entry in the test set
		# save the row number in a different variable
		for rowNum, row in enumerate(rows):
			clusterResults = {key: [] for key in range(len(clusterCenters))}
			for sampleNum, sample in enumerate(clusterCenters):
				res = 0
				for index in range(len(row)):
					res += np.square(np.sqrt(np.square(row[index] - sample[index])))
				clusterResults[sampleNum] = res
			minDistance = min(clusterResults, key=clusterResults.get)
			if frequentClassPerCluster[minDistance] == labels[rowNum]:
				correct+=1
			confusionMatrix[int(labels[rowNum])][int(frequentClassPerCluster[minDistance])]+=1
		print(frequentClassPerCluster)
		print(pd.DataFrame(confusionMatrix))
		print('accuracy=', correct/total)

		for clusterNum, centerPoint in enumerate(clusterCenters):
			imagePixels = copy.deepcopy(centerPoint)
			imagePixels = imagePixels.reshape(8,8)
			plt.imshow(imagePixels, cmap="gray")
			# plt.show()
			filename = 'k'+str(self.k)+'-'+str(clusterNum)+'.png'
			plt.savefig(filename)

		writeConfusionMatrixToCSV(self.k, confusionMatrix)

def writeConfusionMatrixToCSV(k, confusionMatrix):
	filename = str(k)+'_k_units_confusion_matrix.csv'
	with open(filename, 'w') as write:
		write = csv.writer(write)
		write.writerows(confusionMatrix)

# loads the dataset file elements as floats into a 2D numpy array
def loadSet(filename):

	# for our experiment, all the datasets attributes
	# are separated by a comma
	return np.loadtxt(filename, delimiter=',', dtype=np.float64)

def main():

	trainSetFileName = 'data/optdigits.train'
	testSetFileName = 'data/optdigits.test'

	k = 10
	KMC = kMeansClustering(k)

	trainSet = loadSet(trainSetFileName)
	testSet = loadSet(testSetFileName)

	trainResults = KMC.train(trainSet)
	KMC.classify(trainSet, testSet, trainResults['clusterMembership'], trainResults['clusterCenters'])

main()
