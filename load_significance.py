import json
import numpy as np


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

dataset_name = "EMNIST"
train_test_temp_acc_50 = [[0.553703427,0.534247041,0.693109989],
                    [0.557445228,0.508978724,0.656637371],
                    [0.525533319,0.412088871,0.492529958],
                    [0.603954196,0.513213456,0.700366855],
                    [0.425688207,0.460551292,0.505661309],
                    [0.338829696,0.318720967,0.40440017],
                    [0.359393209,0.254218608,0.406451106],
                    [0.38076669,0.330151737,0.273001134],
                    [0.252157331,0.256658018,0.365711421],
                    [0.31967473,0.343527734,0.250466228]]

train_test_temp_acc_10 = [[0.4492715, 0.435319364, 0.562304258],
                        [0.46088773, 0.406711102, 0.523567379],
                        [0.416704983, 0.294336826, 0.323364109],
                        [0.527479947, 0.452155203, 0.631818593],
                        [0.298447341, 0.312179983, 0.327813834],
                        [0.297935218, 0.287463903, 0.376664788],
                        [0.300941706, 0.210251749, 0.359843343],
                        [0.334758103, 0.297321796, 0.220172942],
                        [0.220280096, 0.213854089, 0.32416141],
                        [0.277925044, 0.275287062, 0.156093568]]

train_test_OTDD = [[1571.621826, 1631.72168, 1513.189697],
                [1423.166382, 1526.934082, 1591.773071],
                [1394.848511, 1609.112427, 1589.503784],
                [1500.502075, 1515.185425, 1505.920654],
                [1537.615601, 1525.425293, 1570.547607],
                [1677.11731, 1711.127808, 1663.578857],
                [1453.976074, 1632.437744, 1609.672974],
                [1339.746948, 1389.751099, 1663.830566],
                [1655.865234, 1677.536377, 1731.144775],
                [1516.646362, 1568.089111, 1783.966553]]

train_num, test_num = len(train_test_temp_acc_50), len(train_test_temp_acc_50[0])
json_t = {
    dataset_name: {
        "sub_train_{}".format(train_id): {
            "sub_test_{}".format(test_id): { 
                "epsilon_consume_5.0": train_test_temp_acc_50[train_id][test_id],
                "epsilon_consume_1.0": train_test_temp_acc_10[train_id][test_id]
            } for test_id in range(test_num)
        } for train_id in range(train_num)
    }
}


train_test_OTDD = np.array(train_test_OTDD)
train_test_OTDD = 1 / train_test_OTDD
train_test_OTDD = normalization(train_test_OTDD)
json_k = {
    dataset_name: {
        "sub_train_{}".format(train_id): {
            "sub_test_{}".format(test_id): train_test_OTDD[train_id][test_id] for test_id in range(test_num)
        } for train_id in range(train_num)
    }
}

with open("/home/netlab/DL_lab/opacus_simulation/traces/significance_PBG.json", "w+") as f:
    json.dump(json_t, f)

with open("/home/netlab/DL_lab/opacus_simulation/traces/significance_HIS.json", "w+") as f:
    json.dump(json_k, f)
