# Imports
import csv


# Function that reads the instances from a csv file (every line is a list)
# Receives the name of the file and returns a list of lists with the data (fully converted)
def read_instance(fileName):
    # Open the file
    file = open(fileName, 'r')

    # Declaration of the variable that will store the data
    rawData = []

    # Read line by line
    reader = csv.reader(file)

    # For every line in the file
    for data in reader:
        # If data is empty, skip the line
        if not data:
            continue

        # Declaration of the variable that will store the data of the line (as float)
        row = [float(data[index]) for index in range(len(data) - 2)]

        # Covert the last 2 values to int
        row.extend([int(data[-2]), int(data[-1])])

        # Append the row to the list
        rawData.append(row)


    # Close the file
    file.close()

    return rawData

# Function that reads the data from a csv file (every line is a list)
# Receives the name of the file and returns a list of lists with the data
def read_csv(fileName):
    # Open the file
    file = open(fileName, 'r')

    # Declaration of the variable that will store the data
    rawData = []

    # Read line by line
    reader = csv.reader(file)

    # For every line in the file
    for data in reader:
        # If data is empty, skip the line
        if not data:
            continue

        # Append the row to the list
        rawData.append(data)

    # Close the file
    file.close()

    return rawData

# Function that writes a list of lists to a csv file
# Receives the name of the file and the list of data to write
def write_csv(data, fileName):
    # Open the file
    with open(fileName, "w") as file:
        # Write the data
        writer = csv.writer(file)

        # For every line in the data
        for line in data:
            # Write the line
            writer.writerow(line)
