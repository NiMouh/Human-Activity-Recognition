# Imports
from managefiles import write_csv

# Declaration of the variable that represents the number of the samples
numberOfSamples = 20  # Hz

# Declaration of the Activity labels
activityLabels = {
    'Downstairs': 0,
    'Jogging': 1,
    'Sitting': 2,
    'Standing': 3,
    'Upstairs': 4,
    'Walking': 5
}


# Function "create_instance", it will create an instance with 20 samples of the same activity and same ID
# Every line received have the following formate: "ID,Activity, timestamp, x, y, z"
# Receives a list of lines and returns a list of instances
def create_instances(rawData):
    # Variable that represents the instances
    instances = []

    # Run through the data, if you find a line with the same ID and activity, add it to the instance.
    # Repeat the process until you got 'number_of_samples' samples
    for rowIndex in range(len(rawData)):

        # If the index plus the number of samples overlaps the length of the raw_data, break the loop
        if rowIndex + numberOfSamples > len(rawData):
            break

        # If the next 'number_of_samples' lines doesn't have the same ID and activity, do nothing
        if rawData[rowIndex][0] != rawData[rowIndex + numberOfSamples - 1][0] and rawData[rowIndex][1] != \
                rawData[rowIndex + numberOfSamples - 1][1]:
            continue

        # Initial data structure
        currentInstance = []

        # Insert the id and activity
        for j in range(rowIndex, rowIndex + numberOfSamples):
            # Append the last 3 values of the line to the instance (as float)
            currentInstance.append(float(rawData[j][3]))
            currentInstance.append(float(rawData[j][4]))
            currentInstance.append(float(rawData[j][5]))

        # Append the ID and the activity label to the instance
        currentInstance.append(int(rawData[rowIndex][0]))
        currentInstance.append(int(activityLabels[rawData[rowIndex][1]]))

        # Append the instance to the list of instances
        instances.append(currentInstance)


    # Save the instances to a csv file
    write_csv(instances, "instances.csv")
