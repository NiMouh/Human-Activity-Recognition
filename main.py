# Pratical Project 2: Human Activity Recognition
# Imports
import csv
import math
import random
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve,accuracy_score

# Declaration of the variable that represents the number of the samples
numberOfSamples = 20 # Hz

# Declaration of the Activity labels
activityLabels = {
    'Downstairs': 0,
    'Jogging': 1,
    'Sitting': 2,
    'Standing': 3,
    'Upstairs': 4,
    'Walking': 5
}

# Declaration of the list of instances
instances = []


# Function that reads the data from a csv file
# Receives the name of the file and returns a list of lines
def read_csv(fileName):
    reader = csv.reader(open(fileName, 'r'))

    # Initial data structure
    rawData = []

    # Discard the first line
    next(reader)

    # For each row
    for row in reader:
        rawData.append(row)

    return rawData


# Function that writes data to a csv file
# Receives the name of the file and the data to be written
def write_csv(data, fileName):
    with open(fileName, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(data)


# Function "create_instances", it will create an instance with 20 samples of the same activity and same ID
# Every line received have the following formate: "ID,Activity, timestamp, x, y, z"
# Receives a list of lines and returns a list of instances
def create_instances(rawData):
    # Run through the data, if you find a line with the same ID and activity, add it to the instance.
    # Repeat the process until you got 'number_of_samples' samples
    for rowIndex in range(len(rawData)):

        # If the index plus the number of samples overlaps the length of the raw_data, break the loop
        if rowIndex + numberOfSamples > len(rawData):
            break

        # If the next 'number_of_samples' lines have the same ID and activity, add them to the instance
        if rawData[rowIndex][0] == rawData[rowIndex + numberOfSamples - 1][0] and rawData[rowIndex][1] == \
                rawData[rowIndex + numberOfSamples - 1][1]:

            # Initial data structure
            currentInstance = []

            # Insert the id and activity
            for nextSample in range(numberOfSamples):
                # Append the last 3 values of the line to the instance (as float)
                currentInstance.append(float(rawData[rowIndex + numberOfSamples][3]))
                currentInstance.append(float(rawData[rowIndex + numberOfSamples][4]))
                currentInstance.append(float(rawData[rowIndex + numberOfSamples][5]))

            # Append the ID and the activity label to the instance
            currentInstance.append(rawData[rowIndex][0])
            currentInstance.append(int(activityLabels[rawData[rowIndex][1]]))

            # Append the instance to the list of instances
            instances.append(currentInstance)
            print("Instances created: " + str(len(instances)))

    return instances


# Function that makes K fold cross validation
# It will create 'k' folds, normalize them and write them to a csv file
# Receives the 'k' value that represents the number of folds, and returns 'k' * 2 files with the training and test data
def create_k_fold_validation(k):
    # Declaration of the variable for the folds
    folds = []

    # Declaration of auxiliary variable for the instances
    instancesCreated = instances

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
            # Feedback for the folds and Choosen ID
            print("Fold: " + str(i) + " Tem este ID: " + str(ID))
            # for each instance
            for instance in instancesCreated:
                # If the ID is the same as the instance ID
                if ID == int(instance[0]):
                    # Append the instance to the fold
                    OnGoingFold.append(instance)
            # Feedback to the removed ID from the list
            print("Este ID: " + str(ID) + " foi removido da lista de IDs")
            # Remove the ID from the list of not taken IDs
            idsNotTaken.remove(ID)

        # Append the fold to the list of folds
        folds.append(OnGoingFold)

    # Print how many IDs are left
    print("IDs left: " + str(len(idsNotTaken)))

    # For each ID that is not taken, assign it to the fold with the least instances (to balance the folds)
    for ID in idsNotTaken:
        # Feedback for the ID
        print("Este ID: " + str(ID) + " não foi escolhido para nenhum fold.")

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
        for instance in instancesCreated:
            if ID == int(instance[0]):
                folds[leastIstancesFold].append(instance)

    # For each iteration, create 'k' training and test sets (every fold will be a test set once)
    for indexFold in range(k):
        # Declaration of the variable that represents a copy of the list of folds
        foldsCopy = folds.copy()

        # Declaration of the variable that represents the current test set fold
        print("fold de teste: " + str(indexFold))
        testSet = foldsCopy[indexFold]

        # The rest of the folds are the training set, so we concatenate them
        trainingSet = []
        for j in range(k):
            # If the index is not the same as the test set index
            if j != indexFold:
                # Append all the instances of the fold to the training set
                for instance in foldsCopy[j]:
                    print("Adicionando uma instancia ao training set")
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
        # Feedback for the min and max values
        print("Min: " + str(minValues))
        print("Max: " + str(maxValues))

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
        # Feedback for the training set normalization
        print("Training set normalizado")

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
        # Feedback for the test set normalization
        print("Test set normalizado")

        # Write the test set to a file
        write_csv(testSet, 'fold_test_' + str(currentFold) + '.csv')

        # Write the training set in a csv file
        write_csv(trainingSet, 'fold_train_' + str(currentFold) + '.csv')


# Function that does the one hot encoding
# Receives the instances of the training set and the test set
# Returns 2 lists with the activity IDs encoded
def OneHotEncoding(trainingSet, testSet):
    # Declaration of the variable that represents the data that will be converted
    activitiesData = (["Downstairs", "Jogging", "Sitting", "Standing", "Upstairs", "Walking"])

    # Integer mapping using LabelEncoder
    labelEncoder = LabelEncoder()
    intEncoded = labelEncoder.fit_transform(activitiesData)
    intEncoded = intEncoded.reshape(len(intEncoded), 1)

    # One hot encoding
    oneHotEncoder = OneHotEncoder(sparse=False)
    oneHotEncoded = oneHotEncoder.fit_transform(intEncoded)

    # Declaration of the variable that represents list of the activities on trainingSet
    activitiesOnTraining = []

    # Declaration of the variable that represents list of the activities on testSet
    activitiesOnTest = []

    # For each instance on trainingSet
    for instance in trainingSet:
        # Append the activity on the list
        activitiesOnTraining.append(oneHotEncoded(int(instance[-1])))

    # For each instance on testSet
    for instance in testSet:
        # Append the activity on the list
        activitiesOnTest.append(oneHotEncoded(int(instance[-1])))

    # Return both activity lists encoded
    return activitiesOnTraining, activitiesOnTest


# Main function
if __name__ == '__main__':
    # Read the csv file
    # data = read_csv('time_series_data_human_activities.csv')

    # Create the instances from the raw data
    # instances = create_instances(data)

    # Send the instances to a csv file
    # write_csv(instances, 'instances.csv')

    # Read the instances from the csv file
    instances = read_csv('instances.csv')

    # Declaration of the variable that represents the number of folds
    numberOfFolds = 10

    # Create the K fold cross validation (k = 10)
    create_k_fold_validation(numberOfFolds)

    # For each fold (both training and test)
    for currentFold in range(numberOfFolds):
        # Read the training set (normalized)
        currentTrainingSet = read_csv('fold_train_' + str(currentFold) + '.csv')

        # Read the test set (normalized)
        currentTestSet = read_csv('fold_test_' + str(currentFold) + '.csv')

        # One hot encoding
        onTrainingActivities, onTestActivities = OneHotEncoding(currentTrainingSet, currentTestSet)

        # Create the MLP Classifier
        NeuralNetwork = MLPClassifier(hidden_layer_sizes=(5,5))

        # Train the MLP Classifier
        NeuralNetwork.fit(currentTrainingSet, onTrainingActivities)

        # Predict the test set
        predictions = NeuralNetwork.predict(currentTestSet)

        # Calculate the ROC (Receiver Operating Characteristic ) curve
        fpr, tpr, thresholds = roc_curve(onTestActivities, predictions)

        # Calculate the accuracy (AUROC)
        accuracy = accuracy_score(onTestActivities, predictions)
