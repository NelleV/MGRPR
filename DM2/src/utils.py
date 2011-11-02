import csv
import numpy as np

def load_data(filename):
    """
    Load some data
    """
    csv_file = csv.reader(open(filename, 'r'), delimiter='\t')
    X = []
    for row in csv_file:
        X.append([float(row[0]), float(row[1])])
    return np.array(X)


