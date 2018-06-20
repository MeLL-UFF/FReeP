# source activate py35
import math
import operator

class CustomKNN():
    def __init__(self, training_data, k=3, weights=[]):
        self.training_data = training_data
        self.k = k
        self.weights = weights

    def euclideanDistance(self, instance1, instance2, length):
        distance = 0
        for x in range(length):
            distance += pow((instance1[x] - instance2[x]), 2)
        return math.sqrt(distance)

    def getNeighbors(self, testInstance):
        distances = []
        length = len(testInstance) - 1
        for x in range(len(self.training_data)):
            dist = self.euclideanDistance(testInstance, self.training_data[x], length)
            distances.append((self.training_data[x], dist))
        index_sorted_distances = [b[0] for b in sorted(enumerate([dist for (feature, dist) in distances]),
                                                       key=operator.itemgetter(1))]
        # distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(self.k):
            neighbors.append(index_sorted_distances[x])
        return neighbors

    def getResponse(self, neighbors):
        classVotes = {}
        for x in range(len(neighbors)):
            response = self.training_data[neighbors[x]][-1]
            weight = 1
            if self.weights:
                weight = self.weights[neighbors[x]]
            if response in classVotes:
                classVotes[response] += 1 * weight
            else:
                classVotes[response] = 1 * weight
        sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
        return sortedVotes[0][0]

    def getAccuracy(self, testSet, predictions):
        correct = 0
        for x in range(len(testSet)):
            if testSet[x][-1] == predictions[x]:
                correct += 1
        return (correct / float(len(testSet))) * 100.0

    def predict(self, testSet):
        predictions = []
        for x in range(len(testSet)):
            neighbors = self.getNeighbors(testSet[x])
            result = self.getResponse(neighbors)
            predictions.append(result)
        return predictions

        # def main():
        #     # prepare data
        #     trainingSet = []
        #     testSet = []
        #     split = 0.67
        #     loadDataset('iris.data', split, trainingSet, testSet)
        #
        #     # generate predictions
        #     predictions = []
        #     k = 3
        #     for x in range(len(testSet)):
        #         neighbors = getNeighbors(trainingSet, testSet[x], k)
        #         result = getResponse(neighbors)
        #         predictions.append(result)
        #         print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
        #     accuracy = getAccuracy(testSet, predictions)
        #     print('Accuracy: ' + repr(accuracy) + '%')
        #
        #
        # main()
