from policies.BasePolicy import Policy
import copy
import random
import numpy as np
import math
from scipy.optimize import linear_sum_assignment
import cvxpy as cp

class HISPolicy(Policy):
    def __init__(self, beta, gamma, delta, only_small, logger):
        super().__init__()
        self._name = 'HISPolicy'
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.only_small = only_small
        self.logger = logger

    def report_state(self):
        self.logger.info("policy name: {}".format(self._name))
        self.logger.info("policy args: beta: {}".format(self.beta))
        self.logger.info("policy args: gamma: {}".format(self.gamma))
        self.logger.info("policy args: delta: {}".format(self.delta))
        self.logger.info("policy args: only_small: {}".format(self.only_small))

    def get_LP_result(self, sign_matrix, datablock_privacy_budget_capacity_list, job_privacy_budget_consume_list, solver='SCS'):
        job_num, datablock_num = sign_matrix.shape[0], sign_matrix.shape[1]
        job_privacy_budget_consume_list = np.array(job_privacy_budget_consume_list)[np.newaxis, :]
        datablock_privacy_budget_capacity_list = np.array(datablock_privacy_budget_capacity_list)[np.newaxis, :]

        matrix_X = cp.Variable((job_num, datablock_num), nonneg=True)
        objective = cp.Maximize(
            cp.sum(cp.multiply(sign_matrix, matrix_X))
        )

        constraints = [
            matrix_X >= 0,
            matrix_X <= 1,
            cp.sum(matrix_X, axis=1) <= 1,
            (job_privacy_budget_consume_list @ matrix_X) <= datablock_privacy_budget_capacity_list
        ]

        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve(solver)
        # print(matrix_X.value)
        if cvxprob.status != "optimal":
            print('WARNING: Allocation returned by policy not optimal!')
        return matrix_X.value

    def get_sign_matrix(self, is_large_job, current_all_job_significances, sub_train_datasetidentifier_2_significance,
                        sub_train_datasetidentifier_2_epsilon_capcity,
                        target_epsilon_require, job_priority_weight):
        temp_index_2_datablock_identifier = {}
        sign_matrix = []
        for job_index, job_priority_weight in enumerate(current_all_job_significances):
            temp = []
            for datablock_index, datablock_identifier in enumerate(sub_train_datasetidentifier_2_epsilon_capcity):
                temp_index_2_datablock_identifier[datablock_index] = datablock_identifier
                if (not self.only_small) and (job_index + 1 < len(current_all_job_significances)):
                    if is_large_job and target_epsilon_require <= self.delta * sub_train_datasetidentifier_2_epsilon_capcity[datablock_identifier]: # 最后一个任务不算进去
                        temp.append(0.0)
                    elif (not is_large_job) and target_epsilon_require > self.delta * sub_train_datasetidentifier_2_epsilon_capcity[datablock_identifier]:
                        temp.append(0.0)
                    else:
                        temp.append(sub_train_datasetidentifier_2_significance[datablock_identifier] * job_priority_weight)
                else:
                    temp.append(sub_train_datasetidentifier_2_significance[datablock_identifier] * job_priority_weight)
            sign_matrix.append(temp)
        sign_matrix = np.array(sign_matrix)
        return sign_matrix, temp_index_2_datablock_identifier

    def get_allocation_for_large(self, history_job_priority_weights, sub_train_datasetidentifier_2_significance,
                                sub_train_datasetidentifier_2_epsilon_remain, sub_train_datasetidentifier_2_epsilon_capcity,
                                index, target_epsilon_require, target_datablock_select_num, job_priority_weight):
        assert target_datablock_select_num == 1
        
        selected_datablock_identifiers = []
        calcu_compare_epsilon = 0.0
        
        current_all_job_significances = copy.deepcopy(history_job_priority_weights) 
        current_all_job_significances.append(job_priority_weight)
        sign_matrix, temp_index_2_datablock_identifier = self.get_sign_matrix(
            True, current_all_job_significances, sub_train_datasetidentifier_2_significance,
            sub_train_datasetidentifier_2_epsilon_capcity, target_epsilon_require, job_priority_weight
        )
        row_inds, col_inds = linear_sum_assignment(sign_matrix)
        if index in row_inds:
            target_datablock_index = col_inds[index]
            target_datablock_identifier = temp_index_2_datablock_identifier[target_datablock_index]
            if target_epsilon_require <= sub_train_datasetidentifier_2_epsilon_remain[target_datablock_identifier]:
                selected_datablock_identifiers.append(target_datablock_identifier)
        
        return selected_datablock_identifiers, calcu_compare_epsilon

    def get_allocation_for_small(self, history_job_priority_weights, history_job_budget_consumes, sub_train_datasetidentifier_2_significance,
                                sub_train_datasetidentifier_2_epsilon_remain, sub_train_datasetidentifier_2_epsilon_capcity,
                                index, target_epsilon_require, target_datablock_select_num, job_priority_weight):
        assert target_datablock_select_num == 1
        
        selected_datablock_identifiers = []
        calcu_compare_epsilon = 0.0
        
        current_all_job_significances = copy.deepcopy(history_job_priority_weights)
        current_all_job_significances.append(job_priority_weight)
        current_all_job_budget_consumes = copy.deepcopy(history_job_budget_consumes)
        current_all_job_budget_consumes.append(target_epsilon_require)
        sign_matrix, temp_index_2_datablock_identifier = self.get_sign_matrix(
            False, current_all_job_significances, sub_train_datasetidentifier_2_significance,
            sub_train_datasetidentifier_2_epsilon_capcity, target_epsilon_require, job_priority_weight
        )
        
        datablock_privacy_budget_capacity_list = np.zeros(shape=sign_matrix.shape[1])
        for temp_index in temp_index_2_datablock_identifier:
            datablock_privacy_budget_capacity_list[temp_index] = sub_train_datasetidentifier_2_epsilon_capcity[temp_index_2_datablock_identifier[temp_index]]
        assign_result_matrix = self.get_LP_result(sign_matrix, datablock_privacy_budget_capacity_list, current_all_job_budget_consumes)
        job_num, datablock_num = sign_matrix.shape[0], sign_matrix.shape[1]
        current_job_probability = assign_result_matrix[-1] # 这里其实相当于算出了一个分数, 如果为了这个分数不被泄露, 可以用指数机制加噪, 该方案被证实为满足DP-差分隐私.
        choose_indexes = []
        waiting_select_indexes = np.array(range(datablock_num + 1))
        current_job_probability = list(current_job_probability)
        current_job_probability.append(1.0 - sum(current_job_probability))
        current_job_probability = [proba if proba > 0.0 else 0.0 for proba in current_job_probability]
        current_job_probability = current_job_probability / sum(current_job_probability)
        null_index = len(current_job_probability) - 1
        temp_result = list(np.random.choice(a=waiting_select_indexes, size=target_datablock_select_num, replace=False, p=current_job_probability))
        if null_index in temp_result:
            choose_indexes = copy.deepcopy(temp_result)
            choose_indexes.remove(null_index)
        else:
            choose_indexes = copy.deepcopy(temp_result)
        for choose_index in choose_indexes:
            datablock_identifier = temp_index_2_datablock_identifier[choose_index]
            if target_epsilon_require <= sub_train_datasetidentifier_2_epsilon_remain[datablock_identifier]:
                selected_datablock_identifiers.append(datablock_identifier)
        return selected_datablock_identifiers, calcu_compare_epsilon

    def get_allocation(self, state):
        target_datablock_select_num = state["target_datablock_select_num"]
        job_arrival_index = state["job_arrival_index"]
        all_job_sequence_num = state["all_job_sequence_num"]
        history_job_priority_weights = state["history_job_priority_weights"]
        history_job_budget_consumes = state["history_job_budget_consumes"]

        sub_train_datasetidentifier_2_significance = state["all_sub_train_datasetidentifier_2_significance"]
        sub_train_datasetidentifier_2_epsilon_remain = state["all_sub_train_datasetidentifier_2_epsilon_remain"]
        sub_train_datasetidentifier_2_epsilon_capcity = state["all_sub_train_datasetidentifier_2_epsilon_capcity"]
        target_epsilon_require = state["target_epsilon_require"]
        job_priority_weight = state["job_priority_weight"]

        assert target_datablock_select_num == 1
        
        if len(history_job_priority_weights) < all_job_sequence_num:
            sample_history_job_priority_weights = history_job_priority_weights
            sample_history_job_budget_consumes = history_job_budget_consumes
        else:
            sample_history_job_priority_weights = random.sample(history_job_priority_weights, all_job_sequence_num-1)
            sample_history_job_budget_consumes = random.sample(history_job_budget_consumes, all_job_sequence_num-1)

        
        if self.only_small:
            if job_arrival_index <= self.beta * all_job_sequence_num:
                selected_datablock_identifiers = []
                calcu_compare_epsilon = 0.0
            else:
                selected_datablock_identifiers, \
                    calcu_compare_epsilon = self.get_allocation_for_small(sample_history_job_priority_weights, sample_history_job_budget_consumes, sub_train_datasetidentifier_2_significance,
                                    sub_train_datasetidentifier_2_epsilon_remain, sub_train_datasetidentifier_2_epsilon_capcity,
                                    job_arrival_index, target_epsilon_require, target_datablock_select_num, job_priority_weight)
        else:
            if job_arrival_index <= self.beta * all_job_sequence_num:
                selected_datablock_identifiers = []
                calcu_compare_epsilon = 0.0
            elif self.beta * all_job_sequence_num + 1 <= job_arrival_index <= self.gamma * all_job_sequence_num:
                selected_datablock_identifiers, \
                    calcu_compare_epsilon = self.get_allocation_for_large(sample_history_job_priority_weights, sub_train_datasetidentifier_2_significance,
                                    sub_train_datasetidentifier_2_epsilon_remain, sub_train_datasetidentifier_2_epsilon_capcity,
                                    job_arrival_index, target_epsilon_require, target_datablock_select_num, job_priority_weight)
            else:
                selected_datablock_identifiers, \
                    calcu_compare_epsilon = self.get_allocation_for_small(sample_history_job_priority_weights, sample_history_job_budget_consumes, sub_train_datasetidentifier_2_significance,
                                    sub_train_datasetidentifier_2_epsilon_remain, sub_train_datasetidentifier_2_epsilon_capcity,
                                    job_arrival_index, target_epsilon_require, target_datablock_select_num, job_priority_weight)
        return selected_datablock_identifiers, calcu_compare_epsilon