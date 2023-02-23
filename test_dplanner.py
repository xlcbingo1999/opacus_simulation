import math
import numpy as np

lams = [x / 10.0 for x in range(6, 9, 1)]
gammas = range(1, 5, 1)
S_fs = [x / 10.0 for x in range(5, 15, 1)]

for lam in lams:
    for gamma in gammas:
        for S_f in S_fs:
            
            b_start = ((1 - lam) + math.sqrt(1 - lam)) / lam * (gamma / S_f)

            result = np.random.gamma(2, 1/b_start)
            print(result)