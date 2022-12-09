# Pratical Project 2: Human Activity Recognition
# Imports for other functions
from managefiles import *
from makeinstances import *
from makefolds import *
# Imports for the Neural Network
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve,accuracy_score

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
    # Read the original data from the csv file
    data = read_csv('time_series_data_human_activities.csv')

    # Create the instances from the raw data
    # create_instances(data)

    # Read the instances from the csv file
    # instances = read_csv('instances.csv')

    # Declaration of the variable that represents the number of folds
    numberOfFolds = 10

    # Create the K fold cross validation (k = 10)
    # create_k_fold_validation(numberOfFolds)


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
