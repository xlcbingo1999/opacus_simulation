import itertools
import math

gammas = [i / 10 for i in range(0, 10)] # 0.2, 0.3, 0.4, 0.5
lambs = [i / 100 for i in range(1, 100)] # , 0.6, 0.7, 0.8, 0.9
hs = [800] #  0, 25, 50, 100, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000
ns = [500]

args_product_list = [d for d in itertools.product(gammas, lambs, hs, ns)]

def get_cr(gamma, lamb, h, n):
    if h <= n and h + gamma * n <= n:
        if (1 - lamb) == 0 or (gamma * n + h) == 0 or (2 * (n ** 2)) == 0 or (gamma * n + h) == 0:
            return -1000000
        return (1 - gamma - (1 / (1 - lamb)) * (
            (h/n) * math.log((n-1)/(gamma * n + h)) + (h ** 2 - h)/(2 * (n ** 2)) + math.log(n / (gamma * n + h)) + gamma + (h/n) - 1
        ))
    else:
        if (2 * (1 - lamb)) == 0:
            return -1000000
        return (1 - gamma - ((1 - gamma) ** 2) / (2 * (1 - lamb)))

result = {}
for arg in args_product_list:
    gamma, lamb, h, n = arg
    res = get_cr(gamma, lamb, h, n)
    print("gamma: {}, lamb: {}, h: {}, n: {}; result: {}".format(gamma, lamb, h, n, res))
    result["{}-{}-{}-{}".format(gamma, lamb, h, n)] = res

max_val = max(result.values())
ans = []
for m, n in result.items():
    if n == max_val:
        ans.append(m)
print("ans: ", ans)
print("max_val: ", result[ans[0]])