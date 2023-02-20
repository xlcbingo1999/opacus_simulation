import numpy as np
from functools import reduce
from utils.global_functions import normal_counter, add_2_map

def get_u2_multi_det_S_q(selected_datablock_ids, S_matrix, sub_dataset_us):
    if len(selected_datablock_ids) == 0:
        return 0.0
    else:
        subset_us = [sub_dataset_us[id] for id in selected_datablock_ids]
        subset_us = [np.float_power(u, 2) for u in subset_us]
        multi_us = reduce(lambda x, y: x*y, subset_us)
        sub_S_matrix = S_matrix[selected_datablock_ids, :][:, selected_datablock_ids]
        # print("sub_S_matrix: {}".format(sub_S_matrix))
        det_sub_S_matrix = np.linalg.det(sub_S_matrix) # 这个非常小
        result = multi_us * np.abs(det_sub_S_matrix)
        # print("subset_us: {} with multi_us: {} and det_sub_S_matrix: {}".format(subset_us, multi_us, det_sub_S_matrix))
        # print("calculate score one side: {}".format(result))
        return result

def get_profiler_selection_result(origin_label_distribution, sub_train_datasetidentifier_2_label_distribution, target_select_num):
    '''
    只需要一些metadata即可处理
    origin_sub_label_distributions: map {"}
    '''
    origin_label_distribution = normal_counter(origin_label_distribution)
    origin_keys = set(origin_label_distribution.keys())

    sub_dataset_us = []
    sub_label_distribution_list = []
    sub_train_datasetidentifiers = []
    origin_sub_label_distributions = []
    for sub_train_datasetidentifier in sub_train_datasetidentifier_2_label_distribution:
        sub_train_datasetidentifiers.append(sub_train_datasetidentifier)
        sub_label_distribution = sub_train_datasetidentifier_2_label_distribution[sub_train_datasetidentifier]
        origin_sub_label_distributions.append(sub_label_distribution)

        
        sub_label_distribution = normal_counter(sub_label_distribution)
        sub_label_distribution_list.append(sub_label_distribution)

        sub_dataset_label = set(sub_label_distribution.keys())
        inter_with_origin = origin_keys & sub_dataset_label
        comple_with_origin = origin_keys - sub_dataset_label
        # print("inter_with_origin: {}".format(inter_with_origin))
        # print("comple_with_origin: {}".format(comple_with_origin))

        all_labels_distribution_diff = 0.0
        for inter_label in inter_with_origin:
            all_labels_distribution_diff += pow(sub_label_distribution[inter_label] - origin_label_distribution[inter_label], 2)
        for comple_lable in comple_with_origin:
            all_labels_distribution_diff += pow(sub_label_distribution[comple_lable] - origin_label_distribution[comple_lable], 2)
        # print("all_labels_distribution_diff: {}".format(all_labels_distribution_diff))
        # print("sqrt: {}".format(np.sqrt(all_labels_distribution_diff)))
        sub_dataset_u = 2 - np.sqrt(all_labels_distribution_diff)
        sub_dataset_us.append(sub_dataset_u)

        
    # print("sub_dataset_us: {}".format(sub_dataset_us))

    S_matrix = np.ones(shape=(len(origin_sub_label_distributions), len(origin_sub_label_distributions)))
    selected_datablock_ids = []
    final_scores = []
    not_selected_datablock_ids = list(range(len(sub_dataset_us)))
    while len(selected_datablock_ids) < min(target_select_num, len(origin_sub_label_distributions)):
        origin_score = get_u2_multi_det_S_q(selected_datablock_ids, S_matrix, sub_dataset_us)
        final_score_list = {}
        for vid in not_selected_datablock_ids:
            new_datablock_ids = []
            for id in selected_datablock_ids:
                new_datablock_ids.append(id)
            new_datablock_ids.append(vid)
            # new_datablock_ids.sort()
            # print("selected_datablock_id: {}".format(new_datablock_ids))
            new_score = get_u2_multi_det_S_q(new_datablock_ids, S_matrix, sub_dataset_us)
            final_score = new_score - origin_score
            final_score_list[vid] = final_score
        final_scores.append(final_score_list)
        # print("final_score_list: {}".format(final_score_list))
        select_id = max(final_score_list, key=lambda x: final_score_list[x])
        selected_datablock_ids.append(select_id)
        not_selected_datablock_ids.remove(select_id)
        # print("check select_id: {}".format(select_id))
    
    selected_datablock_identifiers = [sub_train_datasetidentifiers[id] for id in selected_datablock_ids]
    not_selected_datablock_identifiers = [sub_train_datasetidentifiers[id] for id in not_selected_datablock_ids]
    if len(selected_datablock_identifiers) <= 0:
        return selected_datablock_identifiers, not_selected_datablock_identifiers, final_scores, {}
    selected_label_distribution = normal_counter(reduce(add_2_map, [origin_sub_label_distributions[id] for id in selected_datablock_ids]))
    return selected_datablock_identifiers, not_selected_datablock_identifiers, final_scores, selected_label_distribution

