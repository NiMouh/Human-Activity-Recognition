# Pratical Project 2: Human Activity Recognition
# Imports for other functions
from managefiles import *
from makeinstances import *
from makefolds import *

# Imports for the Neural Network
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

# Import for the graphs and stats
import matplotlib.pyplot as plt
import statistics
import numpy as np

# Dictionary of the activity labels (reversed)
activityLabelsReversed = {y: x for x, y in activityLabels.items()}

# Dictionary with the min and max and all AUC's of each activity (two lists)
activityValues = {activity: {'min': [], 'max': [], 'all': []} for activity in activityLabelsReversed}


# Function that receives a list and returns the One Hot Encoding of the list
def oneHotEncoding(item):
    # Declaration of the represents the label encoder
    labelEncoder = LabelEncoder()

    # Declaration of the integer encoded
    integerEncoded = labelEncoder.fit_transform(item)
    integerEncoded = integerEncoded.reshape(len(integerEncoded), 1)

    # Declaration of the one hot encoder
    oneHotEncoder = OneHotEncoder(sparse_output=False)

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
    foldersToAnalyze = 2
    averageFinalScores = []

    # Declaration of the list with all AUC scores and the correpondent fpr and tpr per activity and per fold [auc_score, fpr, tpr]
    allAucScores = []

    # For each fold (both training and test)
    for currentFold in range(foldersToAnalyze):

        # Create the MLP Classifier
        NeuralNetwork = MLPClassifier(hidden_layer_sizes=(5, 5), verbose=True, max_iter=100)

        # If it's the not first iteration, load the neural network
        # if currentFold != 0:
        # NeuralNetwork = joblib.load('NeuralNetwork.pkl')

        # Read the training set (normalized)
        currentTrainingSet = read_instance('fold_train_' + str(currentFold) + '.csv')

        # Read the test set (normalized)
        currentTestSet = read_instance('fold_test_' + str(currentFold) + '.csv')

        # Read the validation set (normalized)
        currentValidationSet = read_instance('fold_validation_' + str(currentFold) + '.csv')

        # Get the activities and user ID'S from the training set (last 2 columns of the bidimensional list)
        activitiesOnTraining = [instance[-1] for instance in currentTrainingSet]
        # usersOnTraining = [instance[-2] for instance in currentTrainingSet]

        # Get the activities and user ID's from the test set (last 2 columns of the bidimensional list)
        activitiesOnTest = [instance[-1] for instance in currentTestSet]
        # usersOnTest = [instance[-2] for instance in currentTestSet]

        # Get the activities and user ID's from the validation set (last 2 columns of the bidimensional list)
        activitiesOnValidation = [instance[-1] for instance in currentValidationSet]
        # usersOnValidation = [instance[-2] for instance in currentValidationSet]

        # Delete the activities and ID's from the training, test set and validation (last 2 columns)
        currentTrainingSet = [instance[:-2] for instance in currentTrainingSet]
        currentTestSet = [instance[:-2] for instance in currentTestSet]
        currentValidationSet = [instance[:-2] for instance in currentValidationSet]

        # Declaration of the variable that represents the encoded version of the activities and user ID's on training set
        activitiesOnTrainingEncoded = oneHotEncoding(activitiesOnTraining)
        # usersOnTrainingEncoded = oneHotEncoding(usersOnTraining)

        # Declaration of the variable that represents the encoded version of the activities and user ID's on test set
        activitiesOnTestEncoded = oneHotEncoding(activitiesOnTest)
        # usersOnTestEncoded = oneHotEncoding(usersOnTest)

        # Declaration of the variable that represents the encoded version of the activities and user ID's on validation set
        activitiesOnValidationEncoded = oneHotEncoding(activitiesOnValidation)
        # usersOnValidationEncoded = oneHotEncoding(usersOnValidation)

        # Train the MLP Classifier
        NeuralNetwork.fit(currentTrainingSet, activitiesOnTrainingEncoded)  # , usersOnTrainingEncoded

        # Save the neural network loss curve on a variable
        trainLossCurve = NeuralNetwork.loss_curve_

        # Predict the test set
        predictions_prob = NeuralNetwork.predict_proba(currentTestSet)
        predictions = NeuralNetwork.predict(currentTestSet)

        # Get the ROC curve, AUC and accuracy
        for currentClass in range(6):
            # ROC curve
            fpr, tpr, thresholds = roc_curve(activitiesOnTestEncoded[:, currentClass], predictions_prob[:, currentClass])

            # Draw the ROC curve for each class
            plt.plot(fpr, tpr, label=str(activityLabelsReversed[currentClass]))

            currentAUC = auc(fpr, tpr)
            activityValues[currentClass]['all'].append(currentAUC)
            print("AUC for ativity " + str(activityLabelsReversed[currentClass]) + ": " + str(currentAUC * 100) + "%")

            # If min and max are empty, add the current AUC,the fpr and tpr
            if not activityValues[currentClass]['min'] and not activityValues[currentClass]['max']:
                activityValues[currentClass]['min'].extend([currentAUC, fpr, tpr])
                activityValues[currentClass]['max'].extend([currentAUC, fpr, tpr])
            # If the current AUC is lower than the min, clean the min and replace it
            elif currentAUC < activityValues[currentClass]['min'][0]:
                activityValues[currentClass]['min'] = [currentAUC, fpr, tpr]
            # If the current AUC is higher than the max, clean the max and replace it
            elif currentAUC > activityValues[currentClass]['max'][0]:
                activityValues[currentClass]['max'] = [currentAUC, fpr, tpr]

        # Save the ROC curve
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.savefig('Imagens/ROC' + str(currentFold) + '.png')
        plt.close()
        plt.clf()

        # Save the train and test loss curve
        plt.plot(trainLossCurve, label='Train Loss')
        plt.xlabel('Time/Experience')
        plt.ylabel('Improvement/Learning')
        plt.legend('Train', loc='upper right')
        plt.savefig('Imagens/LearningCurve' + str(currentFold) + '.png')
        plt.close()
        plt.clf()

        # Get and save the confusion matrix
        for currentClass in range(6):
            ConfusionMatrixDisplay.from_predictions(activitiesOnTestEncoded[:, currentClass], predictions[:, currentClass]).plot()
            plt.savefig(
                'Imagens/ConfusionMatrix' + str(currentFold) + '_' + str(activityLabelsReversed[currentClass]) + '.png')
            plt.close()
            plt.clf()

        # Calculate the final Score
        finalScore = roc_auc_score(activitiesOnTestEncoded, predictions_prob)
        print("Final Score: " + str(finalScore * 100) + "%")

        # Append the final score to the list
        averageFinalScores.append(finalScore)

        # Save the neural network model
        joblib.dump(NeuralNetwork, 'neural_network' + str(currentFold) + '.pkl')

    # Calculate the average final score of the neural network and his standard deviation
    averageFinalScore = sum(averageFinalScores) / len(averageFinalScores)
    deviationFinalScore = statistics.stdev(averageFinalScores)

    # Calculate the final answer
    print("Final Answer: %.2f +/- %.2f" % (averageFinalScore, deviationFinalScore))

    # Get the index of the best and worst AUC score (first column of 3), and draw the ROC curve of both
    for currentClass in range(6):
        # Print the average AUC for each class
        print("Average AUC for ativity " + str(activityLabelsReversed[currentClass]) + ": " + str(
            np.mean(activityValues[currentClass]['all'])))

        # Get the fpr and tpr min
        fprMin = activityValues[currentClass]['min'][1]
        tprMin = activityValues[currentClass]['min'][2]

        # Get the fpr and tpr max
        fprMax = activityValues[currentClass]['max'][1]
        tprMax = activityValues[currentClass]['max'][2]

        # Interpolate the ROC curve
        fprMinInterpolated = np.linspace(fprMin.min(), fprMin.max(), 100)
        fprMaxInterpolated = np.linspace(fprMax.min(), fprMax.max(), 100)
        tprMinInterpolated = np.interp(np.linspace(fprMin.min(), fprMin.max(), 100), fprMin, tprMin)
        tprMaxInterpolated = np.interp(np.linspace(fprMax.min(), fprMax.max(), 100), fprMax, tprMax)

        activityValues[currentClass]['min'] = [fprMinInterpolated, tprMinInterpolated]
        activityValues[currentClass]['max'] = [fprMaxInterpolated, tprMaxInterpolated]

        # Draw the ROC curve
        plt.plot(fprMinInterpolated, tprMinInterpolated, label='Min of ' + str(activityLabelsReversed[currentClass]))
        plt.plot(fprMaxInterpolated, tprMaxInterpolated, label='Max of ' + str(activityLabelsReversed[currentClass]))
        plt.fill_between(fprMaxInterpolated, tprMaxInterpolated, tprMinInterpolated, alpha=0.2)

        # Save the ROC curve
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.savefig('Imagens/FinalROC' + str(activityLabelsReversed[currentClass]) + '.png')
        plt.close()
