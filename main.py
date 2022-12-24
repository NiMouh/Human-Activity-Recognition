# Pratical Project 2: Human Activity Recognition
# Imports for other functions
from managefiles import *
from makeinstances import *
from makefolds import *

# Imports for the Neural Network
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, confusion_matrix

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

    # Declaration of the variable that represents a list with the average final scores of the neural network
    averageFinalScores = []

    # Declaration of the variable that represents a boolean if it's the first iteration
    firstIteration = True

    # Create the MLP Classifier
    NeuralNetwork = MLPClassifier(hidden_layer_sizes=(60, 60), verbose=True)

    # For each fold (both training and test)
    for currentFold in range(foldersToAnalyze):

        # If it's the not first iteration
        # if not firstIteration:
            # Load the neural network
            # NeuralNetwork = joblib.load('NeuralNetwork.pkl')

        # Read the training set (normalized)
        currentTrainingSet = read_instance('fold_train_' + str(currentFold) + '.csv')

        # Read the test set (normalized)
        currentTestSet = read_instance('fold_test_' + str(currentFold) + '.csv')

        # Get the activities from the training set (last column of the bidimensional list)
        activitiesOnTraining = [instance[-1] for instance in currentTrainingSet]

        # Get the user ID's from the training set (second to last column of the bidimensional list)
        # usersOnTraining = [instance[-2] for instance in currentTrainingSet]

        # Get the activities from the test set (last column of the bidimensional list)
        activitiesOnTest = [instance[-1] for instance in currentTestSet]

        # Get the user ID's from the test set (second to last column of the bidimensional list)
        # usersOnTest = [instance[-2] for instance in currentTestSet]

        # Delete the activities from the training set (last column of the bidimensional list)
        currentTrainingSet = [instance[:-2] for instance in currentTrainingSet]

        # Delete the activities from the test set (last column of the bidimensional list)
        currentTestSet = [instance[:-2] for instance in currentTestSet]

        # Declaration of the variable that represents the one hot encoder version of the activities on training set
        activitiesOnTrainingEncoded = oneHotEncoding(activitiesOnTraining)

        # Declaration of the variable that represents the one hot encoder version of the user ID's on training set
        # usersOnTrainingEncoded = oneHotEncoding(usersOnTraining)

        # Declaration of the variable that represents the one hot encoder version of the activities on test set
        activitiesOnTestEncoded = oneHotEncoding(activitiesOnTest)

        # Declaration of the variable that represents the one hot encoder version of the user ID's on test set
        # usersOnTestEncoded = oneHotEncoding(usersOnTest)

        # Train the MLP Classifier
        NeuralNetwork.fit(currentTrainingSet, activitiesOnTrainingEncoded)  # , usersOnTrainingEncoded

        # Predict the test set
        predictions = NeuralNetwork.predict(currentTestSet)

        # Run throught each class and calculate the ROC curve
        for currentClass in range(6):
            # Get the probabilities of the current class
            probabilities = [prediction[currentClass] for prediction in predictions]

            # Get the ROC curve with the probabilities and the activities on test set
            fpr, tpr, thresholds = roc_curve(activitiesOnTestEncoded[:, currentClass], probabilities)

            # Get the AUC (Area Under the Curve) for the current class
            currentAUC = auc(fpr, tpr)
            print("AUC for class " + str(currentClass) + ": " + str(currentAUC * 100) + "%")

            # Draw the ROC curve for each class
            plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % currentAUC)

            # Obtain the accuracy score for the current class
            currentAccuracy = accuracy_score(activitiesOnTestEncoded[:, currentClass], probabilities)
            print('Accuracy for class ' + str(currentClass) + ': ' + str(currentAccuracy))

        # Save the ROC curve
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.savefig('ROC' + str(currentFold) + '.png')
        plt.close()

        # Save the learning curve
        plt.plot(NeuralNetwork.loss_curve_)
        plt.xlabel('Time/Experience')
        plt.ylabel('Improvement/Learning')
        plt.legend(('Loss Curve',), loc='upper right')
        plt.savefig('LearningCurve' + str(currentFold) + '.png')
        plt.close()

        # Make the confusion matrix
        confusionMatrix = confusion_matrix(activitiesOnTestEncoded, predictions)
        print("Confusion Matrix:")
        print(confusionMatrix)

        # Calculate the final Score
        finalScore = roc_auc_score(activitiesOnTestEncoded, predictions)
        print("Final Score: " + str(finalScore * 100) + "%")

        # Append the final score to the list
        averageFinalScores.append(finalScore)

        # Save the neural network model
        # joblib.dump(NeuralNetwork, 'neural_network.pkl')

        # If it's the first iteration (first fold)
        firstIteration = False

    # Calculate the average final score
    # averageFinalScore = sum(averageFinalScores) / len(averageFinalScores)

    # Calculate the deviation of the final scores
    # deviationFinalScore = statistics.stdev(averageFinalScores)

    # Calculate the final answer
    # print("Final Answer: " + str(averageFinalScore * 100) + "% +/- " + str(deviationFinalScore * 100) + "%")
