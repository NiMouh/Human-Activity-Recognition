# Imports
import math, random
from managefiles import read_instance, write_csv


# Function that makes K fold cross validation
# It will create 'k' folds, normalize them and write them to a csv file
# Receives the 'k' value that represents the number of folds, and returns 'k' * 2 files with the training and test data
def create_k_fold_validation(k):
    # Declaration of the variable for the folds
    folds = []

    # Declaration of auxiliary variable for the instances
    instances = read_instance('instances.csv')

    # Declaration of the ID's not taken (36 ID's initially)
    idsNotTaken = [i for i in range(1, 37)]

    # Declaration of the variable that represents the number of instances per fold
    idsPerFold = int(len(idsNotTaken) / k)

    # For each fold, assign 'IDs_per_fold' ID's to it and append every instance with that ID to the fold
    for i in range(k):
        # Declaration of the variable that will save the ID's that will enter on the fold
        AtrributedIDs, idsNotTaken = getFoldIDs(idsNotTaken, idsPerFold)

        # Declaration of the fold that will be atributted the instances
        OnGoingFold = atributteInstances(AtrributedIDs, instances)

        # Append the fold to the list of folds
        folds.append(OnGoingFold)

    # For each ID that is not taken, assign it to the fold with the least instances (to balance the folds)
    for ID in idsNotTaken:

        # Declaration of a variable for the fold with the least instances
        leastIstancesFold = 0

        # Declaration of a variable that saves the lenght of the first fold
        leastInstancesFoldLength = len(folds[0])

        # For each fold retrieve the fold with the least instances
        for currentFoldIndex in range(len(folds)):
            # If the current fold has fewer instances than the previous one
            if len(folds[currentFoldIndex]) < leastInstancesFoldLength:
                # Update the fold with the least instances
                leastInstancesFoldLength = len(folds[currentFoldIndex])
                leastIstancesFold = currentFoldIndex

        # For that fold, add the instances with the ID that is not taken
        for instance in instances:
            if ID == int(instance[-2]):
                folds[leastIstancesFold].append(instance)

    # For each iteration, create 'k' training and test sets (every fold will be a test set once)
    for indexFold in range(k):
        # Declaration of the variable that represents a copy of the list of folds
        foldsCopy = folds.copy()

        testSet = foldsCopy[indexFold]

        # The rest of the folds are the training set, so we concatenate them
        trainingSet = []

        for j in range(k):
            # If the index is the same as the test set, do nothing
            if j == indexFold:
                continue

            # Append all the instances of the fold to the training set
            for instance in foldsCopy[j]:
                trainingSet.append(instance)

        # Declaration of the variable that will save the min and max values of the training set
        minValues, maxValues = getMinMax(trainingSet)

        # Normalize the training set and the test set
        trainingSet, testSet = normalizeData(trainingSet, testSet, minValues, maxValues)

        # Write the training set in a csv file
        write_csv(trainingSet, 'fold_train_' + str(indexFold) + '.csv')

        # Write the test set to a file
        write_csv(testSet, 'fold_test_' + str(indexFold) + '.csv')


# Function that will give the ID's of the instances that will enter on the fold
# It will receive a list of the ID's and the number of ID's per fold
# It will return a list of the ID's that will enter on the fold and the new list of ID's
def getFoldIDs(ids, idsPerFold):
    # Declaration of the variable that will save the ID's of the instances that will enter on the fold
    foldIDs = []

    # For each ID that will enter on the fold
    for i in range(idsPerFold):
        # Randomly pick an ID
        ID = random.choice(ids)
        # Append the ID to the list of ID's that will enter on the fold
        foldIDs.append(ID)
        # Remove the ID from the list of ID's that are not taken
        ids.remove(ID)

    # Return the list of ID's that will enter on the fold and the new list of ID's
    return foldIDs, ids


# Function that will atributte the instances to the folds
# It will receive the OnGoingFold, the list of ID's that will enter on the fold and the list of instances
# It will return the OnGoingFold with the instances that will enter on the fold
def atributteInstances(foldIDs, instances):
    # Declaration of the variable that will represent the fold
    OnGoingFold = []

    # For each instance
    for instance in instances:
        # If the ID of the instance is in the list of ID's that will enter on the fold
        if int(instance[-2]) in foldIDs:
            # Append the instance to the OnGoingFold
            OnGoingFold.append(instance)

    # Return the OnGoingFold with the instances that will enter on the fold
    return OnGoingFold


# Function that will find the min and max values of the training set
# It will receive the training set and will return the min and max values
def getMinMax(trainingSet):
    # Declaration of the variable that will save the min and max values of the training set
    minValues = [math.inf, math.inf, math.inf]
    maxValues = [-math.inf, -math.inf, -math.inf]

    # Determine the min and max of the training set
    for instance in trainingSet:
        for index in range(0, len(instance) - 2, 3):

            # Calculate the min and max of the x axis
            if float(instance[index]) < minValues[0]:
                minValues[0] = float(instance[index])
            if float(instance[index]) > maxValues[0]:
                maxValues[0] = float(instance[index])

            # Calculate the min and max of the y axis
            if float(instance[index + 1]) < minValues[1]:
                minValues[1] = float(instance[index + 1])
            if float(instance[index + 1]) > maxValues[1]:
                maxValues[1] = float(instance[index + 1])

            # Calculate the min and max of the z axis
            if float(instance[index + 2]) < minValues[2]:
                minValues[2] = float(instance[index + 2])
            if float(instance[index + 2]) > maxValues[2]:
                maxValues[2] = float(instance[index + 2])

    return minValues, maxValues


# Function that will normalize the data
# It will receive the training set, test set and min and max values, and will return the normalized training set and test set
def normalizeData(trainingSet, testSet, minValues, maxValues):
    # Normalize the training set
    for instance in trainingSet:
        for index in range(0, len(instance) - 2, 3):
            # If max and min are the same, the value will be 0
            if maxValues[0] - minValues[0] == 0 or maxValues[1] - minValues[1] == 0 or maxValues[2] - \
                    minValues[2] == 0:
                # Remove the value from the instance
                instance.pop(index)
                instance.pop(index + 1)
                instance.pop(index + 2)
                continue
            # Normalize all the axis
            instance[index] = str(
                (float(instance[index]) - minValues[0]) / (maxValues[0] - minValues[0]))
            instance[index + 1] = str(
                (float(instance[index + 1]) - minValues[1]) / (maxValues[1] - minValues[1]))
            instance[index + 2] = str(
                (float(instance[index + 2]) - minValues[2]) / (maxValues[2] - minValues[2]))

    # Normalize the test set
    for instance in testSet:
        for index in range(0, len(instance) - 2, 3):
            # If max and min are the same, the value will be 0
            if maxValues[0] - minValues[0] == 0 or maxValues[1] - minValues[1] == 0 or maxValues[2] - \
                    minValues[2] == 0:
                # Remove the value from the instance
                instance.pop(index)
                instance.pop(index + 1)
                instance.pop(index + 2)
                continue

            # Normalize all the axis
            instance[index] = str(
                (float(instance[index]) - minValues[0]) / (maxValues[0] - minValues[0]))
            instance[index + 1] = str(
                (float(instance[index + 1]) - minValues[1]) / (maxValues[1] - minValues[1]))
            instance[index + 2] = str(
                (float(instance[index + 2]) - minValues[2]) / (maxValues[2] - minValues[2]))

    return trainingSet, testSet
