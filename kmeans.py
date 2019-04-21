import numpy
from collections import defaultdict
from copy import deepcopy
class Kmeans:

    def __init__(self,clusters,pointDict,pointList):
        self.clusters = clusters
        self.pointDict = pointDict
        self.pointList = pointList

    def distCalc(self, points, centre):
        square = 0
        for i, x in enumerate(points):
            y = centre[i]
            square += numpy.square(x-y)
        dist = numpy.sqrt(square)
        return dist

    def getCluster(self, point):
        lMin = 0
        # print(point)
        for i in self.clusters:
            centre = self.clusters[i][0]
            distance = self.distCalc(point, centre)
            if distance < self.distCalc(point, self.clusters[lMin][0]):
                lMin = i
        self.clusters[lMin].append(point)


    def sumPoints(self, points):
        #print("sum points")
        # print(points)
        # print(points[0])
        retVal = list(points[0]).copy()
        for i, point in enumerate(points):
            if i == 0:
                continue
            for j, p in enumerate(point):
                retVal[j] += p
        return retVal

    def updateCentroid(self, points):
        length = len(points)
        retVal = self.sumPoints(points)
        for i, p in enumerate(retVal):
            retVal[i] /= length
        return retVal

    def getNewCentroid(self):
        for i in self.clusters:
            points = self.clusters[i][1:]
            newCentroid = self.updateCentroid(points)
            self.clusters[i][0] = newCentroid

    def clearPoints(self):
        #print("here")
        # print(clusters[0])
        for i in self.clusters:
            self.clusters[i] = self.clusters[i][0:1]
            #print(clusters[i][0])

    def calcMean(self, length):
        mean = 0
        for i in self.clusters:
            centre = self.clusters[i][0]
            points = self.clusters[i][1:]
            for point in points:
                distance = self.distCalc(point, centre)
                mean += distance
        mean /= length
        return mean

    def kmEval(self,trainingSetTripletTags,tripletTagRecord):
        cTag = defaultdict(dict)
        matchDict=defaultdict(int)
        propDict=defaultdict(int)

        for i in self.clusters:
            # print("avnoun")
            # print(len(self.clusters[i][1:]))
            cTag[i] = defaultdict(int)
            points = self.clusters[i][1:]
            numPoints = len(points)
            #print(len(self.clusters[i]))
            for point in points:
                pId = self.pointDict[tuple(point)]

                if pId in trainingSetTripletTags:
                    tempList = trainingSetTripletTags[pId]
                    tagx = tripletTagRecord[(tempList[0][0], tempList[1], tempList[0][1])]
                    # print(len(tagx))
                    tagy = tripletTagRecord[(tempList[0][0], tempList[2], tempList[0][1])]
                    # print(len(tagy))
                    for tag in tagx:
                        cTag[i][tag] += 1
                    for tag in tagy:
                        cTag[i][tag] += 1
                else:
                    print(str(pId)+"hatro")

            for tag in cTag[i]:
                count = cTag[i][tag]
                # print(count)
                if count > int(numPoints / 2):
                    # print("matched: "+str(tag))
                    matchDict[tag] = count

            for point in points:
                for tag in matchDict:
                    # print("Proposed:"+str(tag))
                    propDict[tag] += 1

        return((matchDict,propDict))
