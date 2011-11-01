import numpy as np
from matplotlib import pyplot as plt

from utils import load_data
from classification import QDA

X, Y = load_data('classificationA.train')

p, m_1, m_0, S_1, S_0  = QDA(X, Y)

