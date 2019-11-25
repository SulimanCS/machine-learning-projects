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
				print('try:', tryy, '  run num:', runNum)

		bestRun['MSS'] = self.meanSquareSeparation(bestRun['clusterCenters'])
		bestRun['meanEntropy'] = self.meanEntropy(trainSet, bestRun['clusterMembership'])
		print('final values')
		print('ASME', bestRun['AMSE'])
		print('MSS', bestRun['MSS'])
		print('MEAN ENTROPY', bestRun['meanEntropy'])
		return bestRun

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

	def meanEntropy(self, trainSet, clusterMembership):

		# print(trainSet)
		# print(len(trainSet))
		# print(len(trainSet[0]))
		# exit(1)
		print('in mean entropy')
		entropy = {key: 0 for key in clusterMembership}
		for cluster in clusterMembership:
			print('---')
			clusterDistribution = {}
			total=0
			# print(clusterMembership[cluster])
			# print(cluster)
			for index in clusterMembership[cluster]:
				lastColumn = len(trainSet[index]) - 1
				if trainSet[index, lastColumn] not in clusterDistribution:
					clusterDistribution[trainSet[index, lastColumn]] = 1
				else:
					clusterDistribution[trainSet[index, lastColumn]]+=1

				total+=1
			print(clusterDistribution)
			# print('new start')
			res=0
			for i in clusterDistribution:
				# print(clusterDistribution[i])
				# print('final', clusterDistribution[i]/total * np.log2(clusterDistribution[i]/total))
				res+=clusterDistribution[i]/total * np.log2(clusterDistribution[i]/total)
			res = res * -1
			entropy[cluster] = res
		# 	print('new finish')
		# 	print('---')
		# print('entropy final', entropy)
		# print('CM', clusterMembership)
		# print('len of trainset', len(trainSet))
		# fires = 0
		# for c in clusterMembership:
		# 	fires+=len(clusterMembership[c])
		# print('fires', fires)
		# TODO see the slides, calculate MEAN entropy
		# TODO almost done
		meanEntropy = 0
		for cluster in clusterMembership:
			meanEntropy += (len(clusterMembership[cluster])/len(trainSet)) \
			* entropy[cluster]
		print(meanEntropy)
		return meanEntropy

	def classify(self, trainSet, testSet, clusterMembership, clusterCenters):

		# associate each cluster center with the most frequent class it contains
		frequentClassPerCluster = {}
		# print(len(trainSet))
		# print(len(trainSet[0]))
		lastIndex = len(trainSet[0]) - 1
		for cluster in clusterMembership:
			# print(clusterMembership[cluster])
			clusterLabels = {}
			for index in clusterMembership[cluster]:
				classs = trainSet[index, lastIndex]
				if classs not in clusterLabels:
					clusterLabels[classs]=1
				else:
					clusterLabels[classs]+=1

				# print(classs)
			# print(clusterLabels)
			# print(max(clusterLabels, key=clusterLabels.get))
			frequentClassPerCluster[cluster] = int(max(clusterLabels, key=clusterLabels.get))
			# exit(1)
		# print('final')
		# print(frequentClassPerCluster)

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

