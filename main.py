# Pratical Project 2: Human Activity Recognition
# Imports
import csv
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
            print("Instance created: " + str(len(instances)))

    return instances


# Function that makes K fold cross validation
def create_k_fold_validation(k):
    # Declaration of auxiliary variable
    k_fold_validation = []

    # Divide the instances in k groups and save them in k_fold_validation (this list will have k lists)
    for i in range(k):
        k_fold_validation.append(instances[i::k])

    # For each group
    for i in range(k):
        # Create an aux with the k_fold_validation elements (Training set)
        training_group = k_fold_validation.copy()
        # Create a variable to save the Test set
        test_group = []

        # Choose 2 random instances from the group, add them to the fold_test and remove them from the group (pop)
        for j in range(2):
            random_index = random.randint(0, len(training_group) - 1)
            test_group.append(training_group.pop(random_index))

        # Finally save them in a csv file (separatly)
        write_csv(test_group, 'fold_test_' + str(i) + '.csv')
        write_csv(training_group, 'fold_train_' + str(i) + '.csv')


# Main function
if __name__ == '__main__':
    # Read the csv file
    data = read_csv('time_series_data_human_activities.csv')

    # Create the instances
    instances = create_instances(data)

    # Send the instances to a csv file
    # write_csv(instances, 'instances.csv')

    # Create the K fold cross validation (k = 10)
    create_k_fold_validation(10)

