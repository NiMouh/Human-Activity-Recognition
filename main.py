# Pratical Project 2: Human Activity Recognition

# Declaration of the variable that represents the number of the samples
number_of_samples = 20


# Function that receives the name of the file and returns the data of the csv file
def read_csv_file(file_name):
    file = open(file_name, 'r')
    data = file.read()
    file.close()
    return data


# Every line received have the following formate: "ID,Activity, timestamp, x, y, z"
# This function will create an instance with 20 samples of the same activity and same ID
def create_instance(data, id, activity):
    instance = []
    for i in range(number_of_samples):
        instance.append(data[id][activity][i])
    return instance


# Main function
if __name__ == '__main__':
    # Read the csv file
    data = read_csv_file('time_series_data_human_activities.csv')

    # Split the data in lines
    data = data.split('\n')

    # Remove the first line of the data
    data.pop(0)

    # Create a dictionary that will have the following structure:
    # {ID: {Activity: [samples]}}
    data = {i: {} for i in range(1, 11)}
    for line in data:
        line = line.split(',')
        if line[1] not in data[int(line[0])]:
            data[int(line[0])][line[1]] = []
        data[int(line[0])][line[1]].append([float(line[3]), float(line[4]), float(line[5])])

    # Create the instances (create_instance function)
    instances = []
    for id in data:
        for activity in data[id]:
            instances.append(create_instance(data, id, activity))

    # Print the instances
    print(instances)
