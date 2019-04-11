import sys
sys.path.append("..")
from geneticAlgorithm.solver import *
import numpy as np
chromosome = solve(np.mat([1, 1, 1, 1, 0]).tolist()[0], 5, binCapacity=2)
position = chrom2position(chromosome)
print(position)