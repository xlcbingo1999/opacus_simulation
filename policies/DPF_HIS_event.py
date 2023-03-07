from policies.BasePolicy import Policy
import copy
import random
import numpy as np
import math
import cvxpy as cp
import heapq

class WaitingJob(object):
    def __init__(self, job_id, target_dataset_name, target_datablock_select_num, target_epsilon_require, job_priority_weight):
        self.job_id = job_id
        self.target_dataset_name = target_dataset_name
        self.target_datablock_select_num = target_datablock_select_num
        self.target_epsilon_require = target_epsilon_require
        self.job_priority_weight = job_priority_weight
        self.dominant_share = 0.0

class DPFHISPolicy(Policy):
    def __init__(self, beta, waiting_queue_capacity, logger):
        super().__init__()
        self._name = 'DPFHISPolicy'
        # 保存一个unlocked的量
        self.datablock_identifier_2_unlocked_capacity = {}
        self.waiting_queue = []
        self.waiting_queue_capacity = waiting_queue_capacity

        self.beta = beta
        self.logger = logger
    
    def report_state(self):
        self.logger.info("policy name: {}".format(self._name))
        self.logger.info("policy args: beta: {}".format(self.beta))
        self.logger.info("policy args: waiting_queue_capacity: {}".format(self.waiting_queue_capacity))

    def get_allocation(self, state):
        job_id_2_target_epsilon_require = state["job_id_2_target_epsilon_require"]
        job_id_2_target_datablock_select_num = state["job_id_2_target_datablock_select_num"]
        job_id_2_job_priority_weight = state["job_id_2_job_priority_weight"]
        job_id_2_target_dataset_name = state["job_id_2_target_dataset_name"]
        all_job_sequence_num = state["all_job_sequence_num"]
        history_job_priority_weights = state["history_job_priority_weights"]
        history_job_budget_consumes = state["history_job_budget_consumes"]

        assert len(job_id_2_target_dataset_name) >= self.waiting_queue_capacity
        set_dataset_name = set(job_id_2_target_dataset_name.values())
        assert len(set_dataset_name) == 1 # 必须保证所有的任务都是针对同一个数据集的
        target_dataset_name = list(set_dataset_name)[0]

        sub_train_datasetidentifier_2_significance = state["current_sub_train_datasetidentifier_2_significance"][target_dataset_name]
        sub_train_datasetidentifier_2_epsilon_remain = state["current_sub_train_datasetidentifier_2_epsilon_remain"][target_dataset_name]
        sub_train_datasetidentifier_2_epsilon_capcity = state["current_sub_train_datasetidentifier_2_epsilon_capcity"][target_dataset_name]
    
        self.on_job_arrival(sub_train_datasetidentifier_2_epsilon_capcity, all_job_sequence_num) # OK
        for job_id, target_dataset_name in job_id_2_target_dataset_name.items():
            target_datablock_select_num = job_id_2_target_datablock_select_num[job_id]
            target_epsilon_require = job_id_2_target_epsilon_require[job_id]
            job_priority_weight = job_id_2_job_priority_weight[job_id]
            waiting_job = WaitingJob(job_id, target_dataset_name, target_datablock_select_num, target_epsilon_require, job_priority_weight)
            self.waiting_queue.append(waiting_job)
        job_2_selected_datablock_identifiers = {} 
        calcu_compare_epsilon = 0.0
            # 根据dominant_share排序等待任务
        job_2_selected_datablock_identifiers, calcu_compare_epsilon = self.on_scheduler_time(
            all_job_sequence_num,
            history_job_priority_weights, 
            history_job_budget_consumes,
            sub_train_datasetidentifier_2_significance,
            sub_train_datasetidentifier_2_epsilon_remain,
            sub_train_datasetidentifier_2_epsilon_capcity
        )
        self.waiting_queue = []
        return job_2_selected_datablock_identifiers, calcu_compare_epsilon

    def on_scheduler_time(self, 
                        all_job_sequence_num,
                        history_job_priority_weights, 
                        history_job_budget_consumes,
                        sub_train_datasetidentifier_2_significance,
                        sub_train_datasetidentifier_2_epsilon_remain,
                        sub_train_datasetidentifier_2_epsilon_capcity):
        for job in self.waiting_queue:
            dominant_share = self.get_dominant_share(
                job.target_epsilon_require,
                sub_train_datasetidentifier_2_epsilon_capcity,   
            )
            job.dominant_share = dominant_share
        job_2_selected_datablock_identifiers, \
            calcu_compare_epsilon = self.on_assignment_for_waiting_jobs(all_job_sequence_num,
                                                                        history_job_priority_weights, 
                                                                        history_job_budget_consumes,
                                                                        sub_train_datasetidentifier_2_significance,
                                                                        sub_train_datasetidentifier_2_epsilon_remain,
                                                                        sub_train_datasetidentifier_2_epsilon_capcity)
        return job_2_selected_datablock_identifiers, calcu_compare_epsilon

    def on_assignment_for_waiting_jobs(self,  
                                    all_job_sequence_num,
                                    history_job_priority_weights, 
                                    history_job_budget_consumes,
                                    sub_train_datasetidentifier_2_significance,
                                    sub_train_datasetidentifier_2_epsilon_remain,
                                    sub_train_datasetidentifier_2_epsilon_capcity):
        if len(history_job_priority_weights) + len(self.waiting_queue) < all_job_sequence_num:
            sample_history_job_priority_weights = history_job_priority_weights
            sample_history_job_budget_consumes = history_job_budget_consumes
        else:
            sample_history_job_priority_weights = random.sample(history_job_priority_weights, all_job_sequence_num-len(self.waiting_queue))
            sample_history_job_budget_consumes = random.sample(history_job_budget_consumes, all_job_sequence_num-len(self.waiting_queue))

        job_2_selected_datablock_identifiers, \
            calcu_compare_epsilon = self.get_allocation_for_small(sample_history_job_priority_weights, 
                                                                sample_history_job_budget_consumes, 
                                                                sub_train_datasetidentifier_2_significance,
                                                                sub_train_datasetidentifier_2_epsilon_remain, 
                                                                sub_train_datasetidentifier_2_epsilon_capcity)
        return job_2_selected_datablock_identifiers, calcu_compare_epsilon


    def get_sign_matrix(self, current_all_job_significances, 
                        sub_train_datasetidentifier_2_significance,
                        sub_train_datasetidentifier_2_epsilon_capcity):
        temp_index_2_datablock_identifier = {}
        sign_matrix = []
        for job_index, job_priority_weight in enumerate(current_all_job_significances):
            temp = []
            for datablock_index, datablock_identifier in enumerate(sub_train_datasetidentifier_2_epsilon_capcity):
                temp_index_2_datablock_identifier[datablock_index] = datablock_identifier
                temp.append(sub_train_datasetidentifier_2_significance[datablock_identifier] * job_priority_weight)
            sign_matrix.append(temp)
        sign_matrix = np.array(sign_matrix)
        return sign_matrix, temp_index_2_datablock_identifier

    def get_LP_result(self, sign_matrix, datablock_privacy_budget_capacity_list, job_privacy_budget_consume_list, solver=cp.ECOS):
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
    
    def get_schedule_order(self, waiting_job_selected_sign):
        job_num = len(waiting_job_selected_sign)
        temp_sign_list = [0] * job_num
        heap = []
        for i in range(job_num):
            temp_sign_list[i] = len(waiting_job_selected_sign[i])
            if temp_sign_list[i] > 0:
                for j in range(temp_sign_list[i]):
                    heapq.heappush(heap, (-waiting_job_selected_sign[i][j], (i, j)))
        schedule_order = []
        while len(heap) > 0:
            _, (x, y) = heapq.heappop(heap)
            schedule_order.append((x, y))
        return schedule_order

    def get_allocation_for_small(self, history_job_priority_weights, history_job_budget_consumes, sub_train_datasetidentifier_2_significance,
                                sub_train_datasetidentifier_2_epsilon_remain, sub_train_datasetidentifier_2_epsilon_capcity):
        # assert target_datablock_select_num == 1
        calcu_compare_epsilon = 0.0
        
        current_all_job_significances = copy.deepcopy(history_job_priority_weights)
        current_all_job_budget_consumes = copy.deepcopy(history_job_budget_consumes)
        for job in self.waiting_queue:
            current_all_job_significances.append(job.job_priority_weight)
            current_all_job_budget_consumes.append(job.target_epsilon_require)
        sign_matrix, temp_index_2_datablock_identifier = self.get_sign_matrix(
            current_all_job_significances, sub_train_datasetidentifier_2_significance,
            sub_train_datasetidentifier_2_epsilon_capcity
        )
        
        datablock_privacy_budget_capacity_list = np.zeros(shape=sign_matrix.shape[1])
        for temp_index in temp_index_2_datablock_identifier:
            datablock_privacy_budget_capacity_list[temp_index] = sub_train_datasetidentifier_2_epsilon_capcity[temp_index_2_datablock_identifier[temp_index]]
        assign_result_matrix = self.get_LP_result(sign_matrix, datablock_privacy_budget_capacity_list, current_all_job_budget_consumes)
        job_num, datablock_num = sign_matrix.shape[0], sign_matrix.shape[1]
        waiting_job_selected_datablock_identifiers = []
        waiting_job_selected_sign = []
        for index, job in enumerate(self.waiting_queue):
            temp_selected_datablock_identifiers = []
            temp_selected_probability = []

            current_job_probability = assign_result_matrix[-(len(self.waiting_queue) - index)] # 这里其实相当于算出了一个分数, 如果为了这个分数不被泄露, 可以用指数机制加噪, 该方案被证实为满足DP-差分隐私.
            choose_indexes = []
            waiting_select_indexes = np.array(range(datablock_num + 1))
            current_job_probability = list(current_job_probability)
            current_job_probability.append(1.0 - sum(current_job_probability))
            current_job_probability = [proba if proba > 0.0 else 0.0 for proba in current_job_probability]
            current_job_probability = current_job_probability / sum(current_job_probability)
            null_index = len(current_job_probability) - 1
            result_select_num = job.target_datablock_select_num
            if len(waiting_select_indexes) < job.target_datablock_select_num:
                result_select_num = len(waiting_select_indexes)
            probability_enable_num = sum(p > 0.0 for p in current_job_probability)
            if probability_enable_num < result_select_num:
                result_select_num = probability_enable_num
            temp_result = list(np.random.choice(a=waiting_select_indexes, size=result_select_num, replace=False, p=current_job_probability))
            if null_index in temp_result:
                choose_indexes = copy.deepcopy(temp_result)
                choose_indexes.remove(null_index)
            else:
                choose_indexes = copy.deepcopy(temp_result)
            for choose_index in choose_indexes:
                datablock_identifier = temp_index_2_datablock_identifier[choose_index]
                temp_selected_datablock_identifiers.append(datablock_identifier)
                temp_selected_probability.append(current_job_probability[choose_index])
            waiting_job_selected_datablock_identifiers.append(temp_selected_datablock_identifiers)
            waiting_job_selected_sign.append(temp_selected_probability)
        scheduler_order = self.get_schedule_order(waiting_job_selected_sign)
        job_2_selected_datablock_identifiers = []
        for x, y in scheduler_order:
            job = self.waiting_queue[x]
            datablock_identifier = waiting_job_selected_datablock_identifiers[x][y]
            if job.target_epsilon_require <= sub_train_datasetidentifier_2_epsilon_remain[datablock_identifier]:
                job_2_selected_datablock_identifiers.append((job.job_id, datablock_identifier))
                sub_train_datasetidentifier_2_epsilon_remain[datablock_identifier] -= job.target_epsilon_require

        return job_2_selected_datablock_identifiers, calcu_compare_epsilon
            

    def on_job_arrival(self, sub_train_datasetidentifier_2_epsilon_capcity, all_job_sequence_num):
        for dataset_identifier in sub_train_datasetidentifier_2_epsilon_capcity:
            if dataset_identifier in self.datablock_identifier_2_unlocked_capacity:
                self.datablock_identifier_2_unlocked_capacity[dataset_identifier] = min(
                    sub_train_datasetidentifier_2_epsilon_capcity[dataset_identifier],
                    self.datablock_identifier_2_unlocked_capacity[dataset_identifier] + sub_train_datasetidentifier_2_epsilon_capcity[dataset_identifier] / all_job_sequence_num
                )
            else:
                self.datablock_identifier_2_unlocked_capacity[dataset_identifier] = sub_train_datasetidentifier_2_epsilon_capcity[dataset_identifier] / all_job_sequence_num
        
    def get_dominant_share(self, target_epsilon_require, sub_train_datasetidentifier_2_epsilon_capcity):
        max_dominantshare = 0.0
        result_dataset_identifier = ""
        for dataset_identifier, epsilon_capacity in sub_train_datasetidentifier_2_epsilon_capcity.items():
            if target_epsilon_require / epsilon_capacity > max_dominantshare:
                max_dominantshare = target_epsilon_require / epsilon_capacity
                result_dataset_identifier = dataset_identifier
        assert result_dataset_identifier != ""
        return result_dataset_identifier
        
