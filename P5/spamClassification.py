import sklearn
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import copy
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# loads the dataset file elements as floats into a 2D numpy array
def loadSet(filename):

	# for our experiment, all the datasets attributes
	# are separated by a comma
	return np.loadtxt(filename, delimiter=',', dtype=np.float64)

def preprocessing(data):

	# setup the data

	# total length of dataset
	totalLen = len(data)

	# index of last element
	lastElement = len(data[0]) - 1

	# get all indices where row is not a spam (0)
	zeroesIndices = np.where(data[:, lastElement] == 0)[0]

	# get all indices where row is a spam (1)
	onesIndices = np.where(data[:, lastElement] == 1)[0]

	# store all non spam rows in a separate numpy array
	zeroes = copy.deepcopy(data[zeroesIndices, :])

	# store all spam rows in a separate numpy array
	ones = copy.deepcopy(data[onesIndices, :])

	# split train and test set 50% 50%
	# with equal amount of 0s and 1s

	# initialize train set
	trainSet = np.empty((0, len(data[0])), dtype=np.float64)

	# stack half of 0s and 1s into the train set
	trainSet = np.vstack((trainSet, zeroes[0:len(zeroes)//2, :]))
	trainSet = np.vstack((trainSet, ones[0:len(ones)//2, :]))

	# initialize test set
	testSet = np.empty((0, len(data[0])), dtype=np.float64)

	# stack half of 0s and 1s into the test set
	testSet = np.vstack((testSet, zeroes[len(zeroes)//2:len(zeroes), :]))
	testSet = np.vstack((testSet, ones[len(ones)//2:len(ones), :]))

	# extract labels for both sets
	trainLabels = trainSet[:, -1]
	testLabels = testSet[:, -1]

	# both sets ignore the class column
	trainSet= trainSet[:, :-1]
	testSet= testSet[:, :-1]

	# total length of train and test set should
	# be equal to total dataset length
	assert(len(trainSet) + len(testSet) == len(data))

	# initialize a dictionary that computes mean and std
	# values for every column
	columnData = {key: {} for key in range(len(trainSet[0]))}
	for i in columnData:
		# for every column, compute mean and std
		columnData[i]['mean'] = np.mean(trainSet[:, i])
		columnData[i]['std'] = np.std(trainSet[:, i])

	# scale train data using sklearn
	scaledTrainSet = sklearn.preprocessing.scale(trainSet)

	# initialize scaled test set to be contiguous in memory
	scaledTestSet = np.zeros((len(testSet), len(testSet[0])))

	for rowNum, i in enumerate(testSet):
		for j in range(len(i)):
			# mean = np.mean(trainSet[:, j])
			# std = np.std(trainSet[:, j])
			mean = columnData[j]['mean']
			std = columnData[j]['std']
			scaledTestSet[rowNum, j] = (testSet[rowNum, j] - mean)/std

	return scaledTrainSet, scaledTestSet, trainLabels, testLabels

def linearSVM(data):

	trainSet, testSet, trainLabels, testLabels = preprocessing(data)
	SVClassifier = SVC(kernel='linear')
	# print(s)
	SVClassifier.fit(trainSet, trainLabels)
	predictions = SVClassifier.predict(testSet)

	total = len(testSet)
	correct = 0
	confusionMatrix = confusion_matrix(testLabels, predictions)

	for i in range(len(predictions)):
		if testLabels[i] == predictions[i]: correct+=1

	print(correct/total)
	print(confusionMatrix)
	print(classification_report(testLabels, predictions))

	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	score = SVClassifier.fit(trainSet, trainLabels).decision_function(testSet)
	for i in range(len([0,1])):
		fpr[i], tpr[i], _  = roc_curve(testLabels, score)
		fpr[i], tpr[i], _  = roc_curve(testLabels, predictions)
		roc_auc[i] = auc(fpr[i], tpr[i])

	plt.figure()
	plt.plot(fpr[1], tpr[1])
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.show()

