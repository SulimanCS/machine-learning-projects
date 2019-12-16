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
			rows = trainSet[seperator, :]

			# remove the last column (since it is the class value column)
			rows = rows[:,:-1]

			# store the mean and standard deviation
			# of all columns
			mean = rows.mean(axis=0)
			std = rows.std(axis=0)

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
				'percentage': len(rows)/len(trainSet)
			}

	# displays the mean and std values of every attribute in every class
	def trainingOutput(self):

		for i in self.uniqueClasses:
			for j in range(len(self.uniqueClasses[i]['mean'])):
				print('Class %d, attribute %d, mean = %.2f, std = %.2f' \
					% (i, j+1, self.uniqueClasses[i]['mean'][j], self.uniqueClasses[i]['std'][j]))

	# calculates accuracy based on the training
	# and displays information about every object
	# in the test file, alongside the total
	# classification accuracy
	def classify(self, testSet):

		accuracy = 0
		total = len(testSet)
		confusionMatrix = [[0, 0], [0, 0]]
		predictedClasses = []
		FP = FN = TP = TN = 0

		# main loop that goes over every row of the test set
		for objectID, i in enumerate(testSet, start=1):
			predictions = {}
			# fetch the row without the true class column
			row = i[:-1]
			# fetch the true class type
			label = i[-1]

			# go over all unique classes of the set,
			# compute PDF values for all attributes
			for j in self.uniqueClasses.keys():
				results = []

				# loop over every attribute value in a given entry
				# in the test data, compute PDF for every column
				for z, k in enumerate(row):
					curMean = self.uniqueClasses[j]['mean'][z]
					curStd = self.uniqueClasses[j]['std'][z]
					curStd+=0.0001
					PDF = 1 / (np.sqrt(2*np.pi) * curStd) \
					* np.exp((-1*np.square(k - curMean)) / (2*np.square(curStd)))

					# store all PDF values in a list
					results.append(PDF)

				# add P(class) before computing all values via log addition
				results.append(self.uniqueClasses[j]['percentage'])

				# compute the final prediction value of the current class,
				# for the current entry in the test dataset
				finalPredictionValue = 0
				logProd = 1
				for x in results:
					if x == 0:
						continue
					logProd = logProd * x

				finalPredictionValue = np.log(logProd)
				# store and link the final prediction value to its respective class
				predictions[j] = finalPredictionValue

			predictionMax = max(predictions, key=predictions.get)

			keys = [k for k, v in predictions.items() if v == predictions[predictionMax]]

			# accuracy. This is defined as follows:
			# 1) If there were no ties in your classification result,
			# and the predicted class is correct, the accuracy is 1.

			# 2) If there were no ties in your classification result,
			# and the predicted class is incorrect, the accuracy is 0.

			# 3) If there were ties in your classification result,
			# and the correct class was one of the classes that tied for best,
			# the accuracy is 1 divided by the number of classes that tied for best.

			# 4) If there were ties in your classification result,
			# and the correct class was NOT one of the classes that tied for best, the accuracy is 0.
			currentAccuracy = None

			if label == predictionMax:
				accuracy += 1
				currentAccuracy = 1
			else:
				currentAccuracy = 0

			if label == 0:
				if predictionMax == 0:
					TN+=1
				if predictionMax == 1:
					FP+=1
			if label == 1:
				if predictionMax == 0:
					FN+=1
				if predictionMax == 1:
					TP+=1
			# confusionMatrix[label][predictionMax]+=1
			predictedClasses.append(predictionMax)
			print('ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f' \
				% (objectID, keys[0], predictions[predictionMax], label, currentAccuracy))

		print('classification accuracy=%6.4f' % (accuracy/total))
		print('recall', TP/(TP+FN))
		print('percision', TP/(TP+FP))
		return np.asarray(predictedClasses, dtype=np.float64)

# loads the dataset file elements as floats into a 2D numpy array
def loadSet(filename):

	# for our experiment, all the datasets attributes
	# are separated by a whitespace
	return np.loadtxt(filename, delimiter=',', dtype=np.float64)

# def main(trainSetFilename, testSetFilename):

# 	# load training & testing sets
# 	trainSet, testSet = loadSet(trainSetFilename), loadSet(testSetFilename)

# 	# get how many attributes there are in the given train set
# 	numAttributes = len(trainSet[0]) - 1

# 	# get the last column (class values) from the train set
# 	lastColumn = trainSet[:, len(trainSet[0])-1]

# 	# get only the unique values (eliminate duplicates)
# 	classes = set(lastColumn)

# 	# get how many classes there are for the given dataset
# 	# and create a naive bayes class with that number of classes,
# 	# classes, and number of attributes in the given set
# 	NB = naiveBayes(len(classes), numAttributes, classes)

# 	NB.train(trainSet)
# 	NB.trainingOutput()
# 	NB.classify(testSet)

# if __name__ == '__main__':


# 	trainSetFilename = 'data/spambase.data'
# 	testSetFilename = 'data/spambase.data'

# 	main(trainSetFilename, testSetFilename)
