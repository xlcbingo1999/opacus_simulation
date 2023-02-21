import math
import numpy as np

lam = 0.75
gamma = 2.5
S_f = 1.0
b_start = ((1 - lam) + math.sqrt(1 - lam)) / lam * (gamma / S_f)

result = np.random.gamma(2, 1/b_start)
print(result)