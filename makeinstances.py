# Imports
from managefiles import write_csv

# Declaration of the variable that represents the number of the samples
numberOfSamples = 20  # Hz

# Declaration of the dictionary of the activity labels
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
# Receives a list of lines and returns a list of instances with the formate: "x,y,z,...,x,y,z,User ID,Activity ID"
def create_instances(rawData):
    # Variable that represents the list of instances
    instances = []

    # Run through the data
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

        # Append the last 3 values of the line to the instance (x,y,z)
        for j in range(rowIndex, rowIndex + numberOfSamples):
            currentInstance.extend([float(rawData[j][index]) for index in range(3, 6)])

        # Append the ID and the activity label to the instance
        currentInstance.extend([int(rawData[rowIndex][0]), int(activityLabels[rawData[rowIndex][1]])])

        # Append the instance to the list of instances
        instances.append(currentInstance)

    # Save the instances to a csv file
    write_csv(instances, "instances.csv")
