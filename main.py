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

        if i + number_of_samples > len(raw_data):
            break

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
def create_k_fold_validation(instances, k):
    # Declaration of the highest id
    highest_id = 36

    # Of the 36 IDs choose k IDs to be the training set
    training_set_ids = []

    if k < highest_id:
        for i in range(k):
            # Choose a random ID
            training_set_ids.append(random.randint(1, highest_id))


# Function K fold cross validation

# Shuffle the dataset randomly.

# Split the dataset into k groups

# For each unique group:
# Take the group as a hold out or test data set
# Take the remaining groups as a training data set
# Fit a model on the training set and evaluate it on the test set
# Retain the evaluation score and discard the model

# Summarize the skill of the model using the sample of model evaluation scores


# Main function
if __name__ == '__main__':
    # Read the csv file
    data = read_csv('time_series_data_human_activities.csv')

    # Create the instances
    instances = create_instances(data)

    # Send the instances to a csv file
    write_csv(instances, 'instances.csv')
