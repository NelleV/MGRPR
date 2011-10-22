import csv
import numpy as np

def load_data(filename):
    """
    Load some data
    """
    csv_file = csv.reader(open(filename, 'r'), delimiter='\t')
    X = []
    y = []
    for row in csv_file:
        X.append([row[0], row[1]])
        y.append([row[2]])
    return np.array(X), np.array(y)


