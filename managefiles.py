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
    for row in reader:
        # If the row is not empty, append it to the data
        if row:
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