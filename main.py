# Pratical Project 2: Human Activity Recognition
# Imports for other functions
from managefiles import *
from makeinstances import *
from makefolds import *

# Imports for the Neural Network
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, confusion_matrix

# Import for the graphs
import matplotlib.pyplot as plt

# Dictionary of the activity labels (reversed)
activityLabelsReversed = {y: x for x, y in activityLabels.items()}


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
    # PART 1 - Read the original data from the csv file, and create the instances.
    # data = read_csv('time_series_data_human_activities.csv')
    # create_instances(data)

    # PART 2 - Read the instances from the csv file and create the folds using the K-Fold Cross Validation
    # instances = read_instance('instances.csv')
    # numberOfFolds = 10
    # create_k_fold_validation(numberOfFolds)

    # PART 3 - Read the folds from the csv files and create the neural network
    foldersToAnalyze = 1
    averageFinalScores = []

    # Create the MLP Classifier
    NeuralNetwork = MLPClassifier(hidden_layer_sizes=(60, 60), verbose=True, max_iter=100)

    # For each fold (both training and test)
    for currentFold in range(foldersToAnalyze):

        # If it's the not first iteration, load the neural network
        # if currentFold != 0:
        # NeuralNetwork = joblib.load('NeuralNetwork.pkl')

        # Read the training set (normalized)
        currentTrainingSet = read_instance('fold_train_' + str(currentFold) + '.csv')

        # Read the test set (normalized)
        currentTestSet = read_instance('fold_test_' + str(currentFold) + '.csv')

        # Get the activities and user ID'S from the training set (last 2 columns of the bidimensional list)
        activitiesOnTraining = [instance[-1] for instance in currentTrainingSet]
        # usersOnTraining = [instance[-2] for instance in currentTrainingSet]

        # Get the activities and user ID's from the test set (last 2 columns of the bidimensional list)
        activitiesOnTest = [instance[-1] for instance in currentTestSet]
        # usersOnTest = [instance[-2] for instance in currentTestSet]

        # Delete the activities and ID's from the training and test set (last 2 columns of the bidimensional list)
        currentTrainingSet = [instance[:-2] for instance in currentTrainingSet]
        currentTestSet = [instance[:-2] for instance in currentTestSet]

        # Declaration of the variable that represents the encoded version of the activities and user ID's on training set
        activitiesOnTrainingEncoded = oneHotEncoding(activitiesOnTraining)
        # usersOnTrainingEncoded = oneHotEncoding(usersOnTraining)

        # Declaration of the variable that represents the encoded version of the activities and user ID's on test set
        activitiesOnTestEncoded = oneHotEncoding(activitiesOnTest)
        # usersOnTestEncoded = oneHotEncoding(usersOnTest)

        # Train the MLP Classifier
        NeuralNetwork.fit(currentTrainingSet, activitiesOnTrainingEncoded)  # , usersOnTrainingEncoded

        # Predict the test set
        predictions = NeuralNetwork.predict(currentTestSet)

        # Get the ROC curve, AUC and accuracy
        for currentClass in range(6):
            # ROC curve
            fpr, tpr, thresholds = roc_curve(activitiesOnTestEncoded[:, currentClass], predictions[:, currentClass])

            # Draw the ROC curve for each class
            plt.plot(fpr, tpr, label=str(activityLabelsReversed[currentClass]))

            currentAUC = auc(fpr, tpr)
            print("AUC for class " + str(activityLabelsReversed[currentClass]) + ": " + str(currentAUC * 100) + "%")

            currentAccuracy = accuracy_score(activitiesOnTestEncoded[:, currentClass], predictions[:, currentClass])
            print('Accuracy for ativity ' + str(activityLabelsReversed[currentClass]) + ': ' + str(currentAccuracy))

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

        # Get the confusion matrix
        for currentClass in range(6):
            # Confusion matrix
            confusionMatrix = confusion_matrix(activitiesOnTestEncoded[:, currentClass], predictions[:, currentClass])
            plt.imshow(confusionMatrix, interpolation='nearest', cmap=plt.cm.Blues)

        # Save the confusion matrix
        plt.colorbar()
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig('ConfusionMatrix' + str(currentFold) + '.png')
        plt.close()

        # Calculate the final Score
        finalScore = roc_auc_score(activitiesOnTestEncoded, predictions)
        print("Final Score: " + str(finalScore * 100) + "%")

        # Append the final score to the list
        averageFinalScores.append(finalScore)

        # Save the neural network model
        # joblib.dump(NeuralNetwork, 'neural_network.pkl')

    # Calculate the average final score of the neural network and his standard deviation
    # averageFinalScore = sum(averageFinalScores) / len(averageFinalScores)
    # deviationFinalScore = statistics.stdev(averageFinalScores)

    # Calculate the final answer
    # print("Final Answer: " + str(averageFinalScore * 100) + "% +/- " + str(deviationFinalScore * 100) + "%")
