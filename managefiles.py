# Imports
import csv


# Function that reads the data from a csv file (every line is a list)
# Receives the name of the file and returns a list of lists
def read_csv(fileName):
    # Open the file
    file = open(fileName, 'r')

    # Declaration of the variable that will store the data
    rawData = []

    # Read line by line
    reader = csv.reader(file)
    for data in reader:
        # If the row is not empty, append it to the data
        if data:
            # Declaration of the variable that will store the data of the line
            row = []
            # Convert all the values to float (except the last)
            for index in range(len(data) - 2):
                row.append(float(data[index]))

            # Covert the last 2 values to int
            row.append(int(data[-2]))
            row.append(int(data[-1]))

            # Append the row to the list
            rawData.append(row)

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
