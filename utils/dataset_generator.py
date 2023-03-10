import random

    

def _generate_epsilon_capacity(rng):
    epsilon = 100.0
    r = rng.uniform(0, 1)
    if 0.7 <= r <= 0.8:
        epsilon = 80.0
    elif 0.8 <= r <= 0.95:
        epsilon = 120.0
    elif 0.95 <= r:
        epsilon = 150.0
    return epsilon

def _generate_delta_capacity(rng):
    return 4e-6

# def _generate_significance(rng):
#     significance = 0.1
#     r = rng.uniform(0, 1)
#     if r >= 0.8:
#         significance = 1.0
#     elif 0.4 <= r < 0.8:
#         significance = 0.5
#     else:
#         significance = 0.1
#     return significance

def generate_subtrain_datablocks(dataset_name, block_num,
                                epsilon_capacity_generator=_generate_epsilon_capacity,
                                delta_capacity_generator=_generate_delta_capacity):
    rng = random.Random()
    
    all_blocks = {}
    for index in range(block_num):
        block_name = "train_sub_{}".format(index)
        all_blocks[block_name] = {
            "train_type": index,
            "dataset_name": dataset_name,
            "epsilon_capacity": epsilon_capacity_generator(rng),
            "delta_capacity": delta_capacity_generator(rng),
        }
    sub_train_result = all_blocks
    test_result = {}
    return sub_train_result, test_result

def generate_all_subtrain_datablocks(dataset_name_2_block_num):
    subtrain_datasets_map = {}
    test_datasets_map = {}
    for dataset_name, block_num in dataset_name_2_block_num.items():
        sub_train_result, test_result = generate_subtrain_datablocks(dataset_name, block_num)
        subtrain_datasets_map[dataset_name] = sub_train_result
        test_datasets_map[dataset_name] = test_result
    return subtrain_datasets_map, test_datasets_map


'''
subtrain_datasets_map = {
        "Home_and_Kitchen": {
            "train_sub_0": {
                "epsilon_capacity": 100.0,
                "delta_capacity": 4e-6,
                "time": 0,
                "significance": 1.0,
                "label_distribution": {
                    "0": 3524,
                    "2": 2491,
                    "1": 2423
                },
            },
            "train_sub_1": {
                "epsilon_capacity": 100.0,
                "delta_capacity": 4e-6,
                "time": 0,
                "significance": 1.0,
                "label_distribution": {
                    "0": 3568,
                    "1": 2476,
                    "2": 2394
                },
            },
            "train_sub_2": {
                "epsilon_capacity": 100.0,
                "delta_capacity": 4e-6,
                "time": 0,
                "significance": 1.0,
                "label_distribution": {
                    "0": 3388,
                    "1": 2647,
                    "2": 2403
                },
            },
             "train_sub_3": {
                "epsilon_capacity": 100.0,
                "delta_capacity": 4e-6,
                "time": 0,
                "significance": 1.0,
                "label_distribution": {
                    "0": 3388,
                    "1": 2647,
                    "2": 2403
                },
            },
             "train_sub_4": {
                "epsilon_capacity": 100.0,
                "delta_capacity": 4e-6,
                "time": 0,
                "significance": 1.0,
                "label_distribution": {
                    "0": 3388,
                    "1": 2647,
                    "2": 2403
                },
            },
             "train_sub_5": {
                "epsilon_capacity": 100.0,
                "delta_capacity": 4e-6,
                "time": 0,
                "significance": 1.0,
                "label_distribution": {
                    "0": 3388,
                    "1": 2647,
                    "2": 2403
                },
            },
             "train_sub_6": {
                "epsilon_capacity": 100.0,
                "delta_capacity": 4e-6,
                "time": 0,
                "significance": 1.0,
                "label_distribution": {
                    "0": 3388,
                    "1": 2647,
                    "2": 2403
                },
            },
             "train_sub_7": {
                "epsilon_capacity": 100.0,
                "delta_capacity": 4e-6,
                "time": 0,
                "significance": 1.0,
                "label_distribution": {
                    "0": 3388,
                    "1": 2647,
                    "2": 2403
                },
            },
             "train_sub_8": {
                "epsilon_capacity": 100.0,
                "delta_capacity": 4e-6,
                "time": 0,
                "significance": 1.0,
                "label_distribution": {
                    "0": 3388,
                    "1": 2647,
                    "2": 2403
                },
            },
             "train_sub_9": {
                "epsilon_capacity": 100.0,
                "delta_capacity": 4e-6,
                "time": 0,
                "significance": 1.0,
                "label_distribution": {
                    "0": 3388,
                    "1": 2647,
                    "2": 2403
                },
            },
        }
    }
    test_datasets_map = {
        "Home_and_Kitchen": {
            "time": 0,
        }
    }
'''