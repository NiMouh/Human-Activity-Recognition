# Pratical Project 2: Human Activity Recognition
# Imports for other functions
from managefiles import *
from makeinstances import *
from makefolds import *

# Imports for the Neural Network
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score

# Import for the plots
import matplotlib.pyplot as plt


# Function that receives a list and returns the One Hot Encoding of the list
def oneHotEncoding(item):
    # Declaration of the represents the label encoder
    labelEncoder = LabelEncoder()

    # Declaration of the integer encoded
    integerEncoded = labelEncoder.fit_transform(item)
    integerEncoded = integerEncoded.reshape(len(integerEncoded), 1)

    # Declaration of the one hot encoder
    oneHotEncoder = OneHotEncoder(sparse=False)

    # Return the one hot encoding
    return oneHotEncoder.fit_transform(integerEncoded)

# Main function
if __name__ == '__main__':
    # Read the original data from the csv file
    # data = read_csv('time_series_data_human_activities.csv')

    # Create the instances from the raw data
    # create_instances(data)

    # Read the instances from the csv file
    # instances = read_instance('instances.csv')

    # Declaration of the variable that represents the number of folds
    # numberOfFolds = 10

    # Create the K fold cross validation (k = 10)
    # create_k_fold_validation(numberOfFolds)

    # Declaration of the variable that represents the number of folders to analyze
    foldersToAnalyze = 1

    # For each fold (both training and test)
    for currentFold in range(foldersToAnalyze):
        # Create the MLP Classifier
        NeuralNetwork = MLPClassifier(hidden_layer_sizes=(60, 60), verbose=True)

        # Read the training set (normalized)
        currentTrainingSet = read_instance('fold_train_' + str(currentFold) + '.csv')

        # Read the test set (normalized)
        currentTestSet = read_instance('fold_test_' + str(currentFold) + '.csv')

        # Shuffle the training set
        random.shuffle(currentTrainingSet)

        # Shuffle the test set
        random.shuffle(currentTestSet)

        # Get the activities from the training set (last column of the bidimensional list)
        activitiesOnTraining = [instance[-1] for instance in currentTrainingSet]

        # Get the activities from the test set (last column of the bidimensional list)
        activitiesOnTest = [instance[-1] for instance in currentTestSet]

        # Delete the activities from the training set (last column of the bidimensional list)
        currentTrainingSet = [instance[:-2] for instance in currentTrainingSet]

        # Delete the activities from the test set (last column of the bidimensional list)
        currentTestSet = [instance[:-2] for instance in currentTestSet]

        # Declaration of the variable that represents the one hot encoder version of the activities on training set
        activitiesOnTrainingEncoded = oneHotEncoding(activitiesOnTraining)

        # Declaration of the variable that represents the one hot encoder version of the activities on test set
        activitiesOnTestEncoded = oneHotEncoding(activitiesOnTest)

        # Train the MLP Classifier
        NeuralNetwork.fit(currentTrainingSet, activitiesOnTrainingEncoded)

        # Predict the test set
        predictions = NeuralNetwork.predict(currentTestSet)

        # Run throught each class and calculate the ROC curve
        for currentClass in range(6):
            # Get the probabilities of the current class
            probabilities = [prediction[currentClass] for prediction in predictions]

            # Get the ROC curve
            fpr, tpr, thresholds = roc_curve(activitiesOnTestEncoded[:, currentClass], probabilities)

            # Get the AUC
            currentAUC = auc(fpr, tpr)

            # Draw the ROC curve for each class
            plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % currentAUC)

            # Obtain the accuracy score for the current class
            currentAccuracy = accuracy_score(activitiesOnTestEncoded[:, currentClass], probabilities)

            # Print the accuracy score for the current class
            print('Accuracy for class ' + str(currentClass) + ': ' + str(currentAccuracy))

            # Print the AUC
            print("AUC for class " + str(currentClass) + ": " + str(currentAUC * 100) + "%")

        # Save the plot
        plt.savefig('ROC.png')

        # Calculate the final Score
        Score = roc_auc_score(activitiesOnTestEncoded, predictions)

        # Print the AUC value
        print("Final Score: " + str(Score * 100) + "%")
