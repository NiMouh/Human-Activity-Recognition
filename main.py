# Pratical Project 2: Human Activity Recognition
# Imports
import csv

# Declaration of the variable that represents the number of the samples
number_of_samples = 20  # Hz

# Declaration of the labels
labels = {
    'Downstairs': '0',
    'Jogging': '1',
    'Sitting': '2',
    'Standing': '3',
    'Upstairs': '4',
    'Walking': '5'
}


# Function that receives the name of the csv file and returns a list of instances
def read_csv(file_name):
    reader = csv.reader(open(file_name, 'r'))

    # Initial data structure
    data = []

    # Discard the first line
    next(reader)

    # For each row
    for row in reader:
        data.append(row)

    return data


# Every line received have the following formate: "ID,Activity, timestamp, x, y, z"
# This function will create an instance with 20 samples of the same activity and same ID
def create_instance(data, id, activity):
    # Initial data structure
    instance = []

    # Run through the data, if you find a line with the same ID and activity, add it to the instance.
    # Repeat the process until you got 'number_of_samples' samples
    for row in data:
        if row[0] == id and row[1] == activity:
            # Only the x, y and z values are relevant
            instance.append([float(row[3]), float(row[4]), float(row[5])])
            if len(instance) == number_of_samples:
                break

    return instance


# Main function
if __name__ == '__main__':
    # Read the csv file
    data = read_csv('time_series_data_human_activities.csv')

    # Print data
    print(data)

    # Create the instance to the ID 36 and the activity "Jogging" (Label 1)
    instance = create_instance(data, '36', 'Downstairs')

    # Print the instance
    print(instance)
