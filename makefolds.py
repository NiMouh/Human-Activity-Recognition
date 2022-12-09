# Imports
import math, random
from managefiles import read_csv, write_csv


# Function that makes K fold cross validation
# It will create 'k' folds, normalize them and write them to a csv file
# Receives the 'k' value that represents the number of folds, and returns 'k' * 2 files with the training and test data
def create_k_fold_validation(k):
    # Declaration of the variable for the folds
    folds = []

    # Declaration of auxiliary variable for the instances
    instances = read_csv('instances.csv')

    # Declaration of the ID's not taken (36 ID's initially)
    idsNotTaken = [i for i in range(1, 37)]

    # Declaration of the variable that represents the number of instances per fold
    idsPerFold = int(len(idsNotTaken) / k)

    # For each fold, assign 'IDs_per_fold' ID's to it and append every instance with that ID to the fold
    for i in range(k):
        # Declaration of the fold
        OnGoingFold = []

        # Take 3 IDs that are not taken
        for j in range(idsPerFold):
            # Randomly pick an ID
            ID = random.choice(idsNotTaken)
            # for each instance
            for instance in instances:
                # If the ID is the same as the instance ID (Second to last value)
                if ID == int(instance[-2]):
                    # Append the instance to the fold
                    OnGoingFold.append(instance)
            # Remove the ID from the list of not taken IDs
            idsNotTaken.remove(ID)

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
            if len(folds[currentFoldIndex]) < leastInstancesFoldLength:
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
            # If the index is not the same as the test set index
            if j != indexFold:
                # Append all the instances of the fold to the training set
                for instance in foldsCopy[j]:
                    trainingSet.append(instance)

        # Declaration of the variable that will save the min and max values of the training set
        minValues = [math.inf, math.inf, math.inf]
        maxValues = [-math.inf, -math.inf, -math.inf]

        # Determine the min and max of the training set
        for instance in trainingSet:
            for index in range(2, len(instance), 3):

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

        # Normalize the training set
        for instance in trainingSet:
            for index in range(2, len(instance), 3):
                # If max and min are the same, the value will be 0
                if maxValues[0] - minValues[0] == 0 or maxValues[1] - minValues[1] == 0 or maxValues[2] - \
                        minValues[2] == 0:
                    # Remove the value from the instance
                    instance.pop(index)
                    instance.pop(index + 1)
                    instance.pop(index + 2)
                else:
                    # Normalize all the axis
                    instance[index] = str(
                        (float(instance[index]) - minValues[0]) / (maxValues[0] - minValues[0]))
                    instance[index + 1] = str(
                        (float(instance[index + 1]) - minValues[1]) / (maxValues[1] - minValues[1]))
                    instance[index + 2] = str(
                        (float(instance[index + 2]) - minValues[2]) / (maxValues[2] - minValues[2]))

        # Write the training set in a csv file
        write_csv(trainingSet, 'fold_train_' + str(indexFold) + '.csv')

        # Normalize the test set
        for instance in testSet:
            for index in range(2, len(instance), 3):
                # If max and min are the same, the value will be 0
                if maxValues[0] - minValues[0] == 0 or maxValues[1] - minValues[1] == 0 or maxValues[2] - \
                        minValues[2] == 0:
                    # Remove the value from the instance
                    instance.pop(index)
                    instance.pop(index + 1)
                    instance.pop(index + 2)
                else:
                    # Normalize all the axis
                    instance[index] = str(
                        (float(instance[index]) - minValues[0]) / (maxValues[0] - minValues[0]))
                    instance[index + 1] = str(
                        (float(instance[index + 1]) - minValues[1]) / (maxValues[1] - minValues[1]))
                    instance[index + 2] = str(
                        (float(instance[index + 2]) - minValues[2]) / (maxValues[2] - minValues[2]))

        # Write the test set to a file
        write_csv(testSet, 'fold_test_' + str(indexFold) + '.csv')
