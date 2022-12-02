# Pratical Project 2: Human Activity Recognition
# Imports
import csv
import math
import random

# Declaration of the variable that represents the number of the samples
number_of_samples = 20  # Hz

# Declaration of the labels
labels = {
    'Downstairs': 0,
    'Jogging': 1,
    'Sitting': 2,
    'Standing': 3,
    'Upstairs': 4,
    'Walking': 5
}

# Declaration of the list of instances
instances = []


# Function that receives the name of the csv file and returns a list of instances
def read_csv(file_name):
    reader = csv.reader(open(file_name, 'r'))

    # Initial data structure
    raw_data = []

    # Discard the first line
    next(reader)

    # For each row
    for row in reader:
        raw_data.append(row)

    return raw_data


# Function that receives the data and write it to a csv file
def write_csv(data, file_name):
    with open(file_name, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(data)


# Every line received have the following formate: "ID,Activity, timestamp, x, y, z"
# This function will create an instance with 20 samples of the same activity and same ID
def create_instances(raw_data):
    # Run through the data, if you find a line with the same ID and activity, add it to the instance.
    # Repeat the process until you got 'number_of_samples' samples
    for i in range(len(raw_data)):

        # If the index plus the number of samples overlaps the length of the raw_data, break the loop
        if i + number_of_samples > len(raw_data):
            break

        # If the next 'number_of_samples' lines have the same ID and activity, add them to the instance
        if raw_data[i][0] == raw_data[i + number_of_samples - 1][0] and raw_data[i][1] == \
                raw_data[i + number_of_samples - 1][1]:

            # Initial data structure
            instance = [raw_data[i][0], labels[raw_data[i][1]]]

            # Insert the id and activity
            for j in range(i, i + number_of_samples):
                # Append the last 3 values of the line to the instance (as float)
                instance.append(float(raw_data[j][3]))
                instance.append(float(raw_data[j][4]))
                instance.append(float(raw_data[j][5]))

            # Append the instance to the list of instances
            instances.append(instance)
            print("Instances created: " + str(len(instances)))

    return instances


# Function that makes K fold cross validation
def create_k_fold_validation(k):
    # Declaration of the variable for the folds
    folds = []

    # Declaration of auxiliary variable for the instances
    aux_instances = instances

    # Declaration of the ID's not taken (36 ID's initially)
    not_taken_IDs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                     28, 29, 30, 31, 32, 33, 34, 35, 36]

    IDs_per_fold = int(len(not_taken_IDs) / k)

    # 1000 instances / 10 folds = 100 instances per fold
    # 36 ID's / 10 folds = 3.6 ID's per fold

    # For each fold
    for i in range(k):
        # Declaration of the fold
        fold = []

        # Take 3 IDs that are not taken
        for j in range(IDs_per_fold):
            # Randomly pick an ID
            ID = random.choice(not_taken_IDs)
            # Feedback for the folds and Choosen ID
            print("Fold: " + str(i) + " Tem este ID: " + str(ID))
            # for each instance
            for instance in aux_instances:
                # If the ID is the same as the instance ID
                if ID == int(instance[0]):
                    # Append the instance to the fold
                    fold.append(instance)
            # Feedback to the removed ID from the list
            print("Este ID: " + str(ID) + " foi removido da lista de IDs")
            # Remove the ID from the list of not taken IDs
            not_taken_IDs.remove(ID)

        # Append the fold to the list of folds
        folds.append(fold)

    # Print how many IDs are left
    print("IDs left: " + str(len(not_taken_IDs)))

    # For each ID that is not taken
    for ID in not_taken_IDs:
        # Feedback for the ID
        print("Este ID: " + str(ID) + " não foi escolhido para nenhum fold.")

        # Declaration of a variable for the fold with the least instances
        fold_with_least_instances = 0

        # Declaration of a variable that saves the lenght of the first fold
        length_of_fold_with_least_instances = len(folds[0])

        # For each fold retrieve the fold with the least instances
        for i in range(len(folds)):
            if len(folds[i]) < length_of_fold_with_least_instances:
                length_of_fold_with_least_instances = len(folds[i])
                fold_with_least_instances = i

        # For that fold, add the instances with the ID that is not taken
        for instance in aux_instances:
            if ID == int(instance[0]):
                folds[fold_with_least_instances].append(instance)

    # For each iteration
    for i in range(k):
        foldsCopy = folds.copy()

        # Choose one fold as the test fold
        print("fold de teste: " + str(i))
        test_set = foldsCopy[i]

        # The rest of the folds are the training set, so we concatenate them
        training_set = []
        for j in range(k):
            if j != i:
                for instance in foldsCopy[j]:
                    print("Adicionando uma instancia ao training set")
                    training_set.append(instance)

        # Declaration of the variable that will save the min and max values of the training set
        min = [math.inf, math.inf, math.inf]
        max = [-math.inf, -math.inf, -math.inf]

        # Determine the min and max of the training set
        for instance in training_set:
            for index in range(2, len(instance), 3):

                # Calculate the min and max of the x axis
                if float(instance[index]) < min[0]:
                    min[0] = float(instance[index])
                if float(instance[index]) > max[0]:
                    max[0] = float(instance[index])

                # Calculate the min and max of the y axis
                if float(instance[index + 1]) < min[1]:
                    min[1] = float(instance[index + 1])
                if float(instance[index + 1]) > max[1]:
                    max[1] = float(instance[index + 1])

                # Calculate the min and max of the z axis
                if float(instance[index + 2]) < min[2]:
                    min[2] = float(instance[index + 2])
                if float(instance[index + 2]) > max[2]:
                    max[2] = float(instance[index + 2])
        # Feedback for the min and max values
        print("Min: " + str(min))
        print("Max: " + str(max))

        # Normalize the training set
        for instance in training_set:
            print("Normalizando o training set")
            for index in range(2, len(instance), 3):
                # If max and min are the same, the value will be 0
                if max[0] - min[0] == 0 or max[1] - min[1] == 0 or max[2] - min[2] == 0:
                    # Remove the value from the instance
                    instance.pop(index)
                    instance.pop(index + 1)
                    instance.pop(index + 2)
                else:
                    instance[index] = str((float(instance[index]) - min[0]) / (max[0] - min[0]))
                    instance[index + 1] = str((float(instance[index + 1]) - min[1]) / (max[1] - min[1]))
                    instance[index + 2] = str((float(instance[index + 2]) - min[2]) / (max[2] - min[2]))
        print("Training set normalizado")
        # Normalize the test set
        for instance in test_set:
            print("Normalizando o test set")
            for index in range(2, len(instance), 3):
                # If max and min are the same, the value will be 0
                if max[0] - min[0] == 0 or max[1] - min[1] == 0 or max[2] - min[2] == 0:
                    # Remove the value from the instance
                    instance.pop(index)
                    instance.pop(index + 1)
                    instance.pop(index + 2)
                else:
                    instance[index] = str((float(instance[index]) - min[0]) / (max[0] - min[0]))
                    instance[index + 1] = str((float(instance[index + 1]) - min[1]) / (max[1] - min[1]))
                    instance[index + 2] = str((float(instance[index + 2]) - min[2]) / (max[2] - min[2]))
        print("Test set normalizado")

        # Finally save them in a csv file (separatly)
        write_csv(test_set, 'fold_test_' + str(i) + '.csv')

        # Write the training set in a csv file
        write_csv(training_set, 'fold_train_' + str(i) + '.csv')


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

    # Create the K fold cross validation (k = 10)
    create_k_fold_validation(10)
