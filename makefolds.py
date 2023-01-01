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
        OnGoingFold = [instance for instance in instances if int(instance[-2]) in AtrributedIDs]

        # Append the fold to the list of folds
        folds.append(OnGoingFold)

    # For each ID that is not taken, assign it to the fold with the least instances (to balance the folds)
    for ID in idsNotTaken:
        leastIstancesFold = folds.index(min(folds, key=len))

        # For each instance, check if the ID is the ID that is not taken and append it to the fold with the least instances
        for instance in instances:
            if int(instance[-2]) == ID:
                folds[leastIstancesFold].append(instance)

    # For each iteration, create 'k' training and test sets (every fold will be a test set once)
    for indexFold in range(k):
        foldsCopy = folds.copy()

        # Declaration of the variable that will get the indexes (both validation and test set)
        validationSetindexes, testSetindexes = getFoldIndexes(k, indexFold)

        # Declaration of the variable that will save the current test set and the training set (2 teste 1 validation 7 treino)
        testSet = []
        trainingSet = []
        validationSet = []

        # For each fold, except the test set fold, concatenate the instances to the training set
        for j in range(k):
            # If the index is the same as the test set, do nothing
            if j == indexFold:
                validationSet.extend(foldsCopy[j])
            # If the index is the same as one of the test set indexes, concatenate the instances to the test set
            elif j in testSetindexes:
                testSet.extend(foldsCopy[j])
            # Else, concatenate the instances to the training set
            else:
                trainingSet.extend(foldsCopy[j])

        # Declaration of the variable that will save the min and max values of the training set
        minValues, maxValues = getMinMax(trainingSet)

        # Normalize the training set and the test set
        trainingSet, testSet, validationSet = normalizeData(trainingSet, testSet, validationSet, minValues, maxValues)

        # Shuffle the training set, the test set and the validation set
        random.shuffle(trainingSet)
        random.shuffle(testSet)
        random.shuffle(validationSet)

        # Write the training, test set and validation set in a csv file
        write_csv(trainingSet, 'fold_train_' + str(indexFold) + '.csv')
        write_csv(testSet, 'fold_test_' + str(indexFold) + '.csv')
        write_csv(validationSet, 'fold_validation_' + str(indexFold) + '.csv')


# Function that will give the ID's of the instances that will enter on the fold
# It will receive a list of the ID's and the number of ID's per fold
# It will return a list of the ID's that will enter on the fold and the new list of ID's
def getFoldIDs(ids, idsPerFold):
    # Declaration of the variable that will save the ID's of the instances that will enter on the fold
    foldIDs = []

    # For each ID that will enter on the fold, randomly choose an ID from the list of ID's and append it to the fold
    for i in range(idsPerFold):
        ID = random.choice(ids)
        foldIDs.append(ids.pop(ids.index(ID)))

    # Return the list of ID's that will enter on the fold and the new list of ID's
    return foldIDs, ids


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
def normalizeData(trainingSet, testSet, validationSet, minValues, maxValues):
    # Normalize the training set
    for instance in trainingSet:
        for index in range(0, len(instance) - 2, 3):
            # If max and min are the same, the value will be removed
            if maxValues[0] - minValues[0] == 0 or maxValues[1] - minValues[1] == 0 or maxValues[2] - \
                    minValues[2] == 0:
                instance.pop(index)
                instance.pop(index + 1)
                instance.pop(index + 2)
                continue
            # Normalize all the axis
            instance[index] = (float(instance[index]) - minValues[0]) / (maxValues[0] - minValues[0])
            instance[index + 1] = (float(instance[index + 1]) - minValues[1]) / (maxValues[1] - minValues[1])
            instance[index + 2] = (float(instance[index + 2]) - minValues[2]) / (maxValues[2] - minValues[2])

    # Normalize the test set
    for instance in testSet:
        for index in range(0, len(instance) - 2, 3):
            # If max and min are the same, the value will be removed
            if maxValues[0] - minValues[0] == 0 or maxValues[1] - minValues[1] == 0 or maxValues[2] - \
                    minValues[2] == 0:
                instance.pop(index)
                instance.pop(index + 1)
                instance.pop(index + 2)
                continue

            # Normalize all the axis
            instance[index] = (float(instance[index]) - minValues[0]) / (maxValues[0] - minValues[0])
            instance[index + 1] = (float(instance[index + 1]) - minValues[1]) / (maxValues[1] - minValues[1])
            instance[index + 2] = (float(instance[index + 2]) - minValues[2]) / (maxValues[2] - minValues[2])

    # Normalize the validation set
    for instance in validationSet:
        for index in range(0, len(instance) - 2, 3):
            # If max and min are the same, the value will be removed
            if maxValues[0] - minValues[0] == 0 or maxValues[1] - minValues[1] == 0 or maxValues[2] - \
                    minValues[2] == 0:
                instance.pop(index)
                instance.pop(index + 1)
                instance.pop(index + 2)
                continue

            # Normalize all the axis
            instance[index] = (float(instance[index]) - minValues[0]) / (maxValues[0] - minValues[0])
            instance[index + 1] = (float(instance[index + 1]) - minValues[1]) / (maxValues[1] - minValues[1])
            instance[index + 2] = (float(instance[index + 2]) - minValues[2]) / (maxValues[2] - minValues[2])

    return trainingSet, testSet, validationSet


# Function that receives the number of folds and makes a list from 0 to the number of folds - 1, and the current index
# And it will return 2 lists, one with size 1 that will be for the validation set (current index)
# one with size 2 that will be for the test set (next 2 indexes, but if the current index + 2 is bigger than the number of folds, it will be the first 2 indexes)
def getFoldIndexes(k, indexFold):
    # Generate all the sequences of indexes with size 3 of the number of folds
    indexes = [[index, (index + 1) % k, (index + 2) % k] for index in range(k)]

    # Select the validation set
    validationSet = indexes[indexFold][0]
    # Select the test set
    testSet = indexes[indexFold][1:]

    return validationSet, testSet
