# Pratical Project 2: Human Activity Recognition
# Imports for other functions
from managefiles import *
from makeinstances import *
from makefolds import *

# Imports for the Neural Network
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_curve, auc, roc_auc_score

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
        NeuralNetwork = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 5), random_state=1,
                                      verbose=True)

        # Read the training set (normalized)
        currentTrainingSet = read_instance('fold_train_' + str(currentFold) + '.csv')

        # Read the test set (normalized)
        currentTestSet = read_instance('fold_test_' + str(currentFold) + '.csv')

        # Get the activities from the training set (last column of the bidimensional list)
        activitiesOnTraining = [instance[-1] for instance in currentTrainingSet]

        # Get the activities from the test set (last column of the bidimensional list)
        activitiesOnTest = [instance[-1] for instance in currentTestSet]

        # Declaration of the variable that represents the label Encoder
        labelEncoder = LabelEncoder()

        # Declaration of the variable that represents the one hot encoder
        oneHotEncoder = OneHotEncoder(sparse=False)

        # Declaration of the variable that represents the one hot encoder version of the activities on training set
        activitiesOnTrainingEncoded = labelEncoder.fit_transform(activitiesOnTraining)

        activitiesOnTrainingEncoded = activitiesOnTrainingEncoded.reshape(len(activitiesOnTrainingEncoded), 1)
        activitiesOnTrainingEncoded = oneHotEncoder.fit_transform(activitiesOnTrainingEncoded)

        # Declaration of the variable that represents the one hot encoder version of the activities on test set
        activitiesOnTestEncoded = labelEncoder.fit_transform(activitiesOnTest)

        activitiesOnTestEncoded = activitiesOnTestEncoded.reshape(len(activitiesOnTestEncoded), 1)
        activitiesOnTestEncoded = oneHotEncoder.fit_transform(activitiesOnTestEncoded)

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

            # Print the AUC
            print("AUC for class " + str(currentClass) + ": " + str(currentAUC * 100) + "%")

        # Calculate the final Score
        Score = roc_auc_score(activitiesOnTestEncoded, predictions)

        # Print the AUC value
        print("Final Score: " + str(Score * 100) + "%")
