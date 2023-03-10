import json
import numpy as np


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

dataset_name = "EMNIST"
model_name = "CNN"
train_test_acc_50 = [[0.642222106,	0.596250892,	0.792394817],	
                    [0.648661733,	0.60965991,	0.759118736],
                    [0.636844158,	0.504989386,	0.650562108],
                	[0.678201616,	0.603115261,	0.783456504],
                	[0.540670037,	0.579032183,	0.666131973],
                	[0.422913283,	0.379158437,	0.4356547],
                	[0.425517291,	0.300134242,	0.496455789],
                	[0.443352014,	0.409564078,	0.384862095],
                	[0.403198808,	0.35518682,	0.440906078],
                	[0.42196995,	0.458325654,	0.475160748]]


train_test_acc_10 = [[0.559392512,	0.508596957,	0.702752709],
                    [0.556985795,	0.50644356,	0.643989742],
                    [0.527880907,	0.408066839,	0.506023526],
                    [0.597628236,	0.512752712,	0.687918901],
                    [0.423083007,	0.478781462,	0.517834663],
                    [0.36437726,	0.328371078,	0.409144849],
                    [0.373941302,	0.259206355,	0.417701989],
                    [0.392590404,	0.341718465,	0.280789793],
                    [0.276449382,	0.254882634,	0.380156428],
                    [0.347175717,	0.336531729,	0.283861488]]


train_num, test_num = len(train_test_acc_50), len(train_test_acc_10[0])
json_t = {
    dataset_name: {
        "sub_train_{}".format(train_id): {
            "sub_test_{}".format(test_id): { 
                model_name: {
                    "epsilon_consume_5.0": train_test_acc_50[train_id][test_id],
                    "epsilon_consume_1.0": train_test_acc_10[train_id][test_id]
                }
            } for test_id in range(test_num)
        } for train_id in range(train_num)
    }
}

with open("/home/netlab/DL_lab/opacus_simulation/traces/accuracy_result.json", "w+") as f:
    json.dump(json_t, f)

