from policies.BasePolicy import Policy
import copy
import random
import numpy as np
import math

class PBGPolicy(Policy):
    def __init__(self, comparison_cost_epsilon, comparison_z_threshold, L, U):
        super().__init__()
        self._name = 'PBGPolicy'
        self.comparison_cost_epsilon = comparison_cost_epsilon
        self.comparison_z_threshold = comparison_z_threshold
        self.L = L
        self.U = U

    def report_state(self, logger):
        logger.info("policy name: {}".format(self._name))
        logger.info("policy args: comparison_cost_epsilon: {}".format(self.comparison_cost_epsilon))
        logger.info("policy args: comparison_z_threshold: {}".format(self.comparison_z_threshold))
        logger.info("policy args: L: {}".format(self.L))
        logger.info("policy args: U: {}".format(self.U))

    def Lap(self, scale):
        return np.random.laplace(loc=0.0, scale=scale)

    def Threshold_func(self, datablock_z):
        if 0.0 <= datablock_z <= self.comparison_z_threshold:
            return self.L
        elif self.comparison_z_threshold < datablock_z <= 1.0:
            return math.pow(self.U * math.exp(1) / self.L, datablock_z) * self.L / math.exp(1)
        return self.L

    def filter_by_threshold(self, datablock_epsilon_capacity, datablock_z, target_epsilon_consume, significance_plus_weight):
        if datablock_z < self.comparison_z_threshold:
            is_select = True
            # new_z = datablock_z + target_epsilon_consume / datablock_epsilon_capacity
            compare_epsilon = 0.0
        else:
            if ((significance_plus_weight / (target_epsilon_consume + self.comparison_cost_epsilon)) + self.Lap(4.0/self.comparison_cost_epsilon) > self.Threshold_func(datablock_z) + self.Lap(2.0/self.comparison_cost_epsilon) 
            and target_epsilon_consume + self.comparison_cost_epsilon < (1.0 - datablock_z) * datablock_epsilon_capacity):
                is_select = True
                # new_z = datablock_z + (target_epsilon_consume + self.comparison_cost_epsilon)/datablock_epsilon_capacity
                compare_epsilon = self.comparison_cost_epsilon
            else:
                is_select = False
                # new_z = datablock_z
                compare_epsilon = 0.0
        return is_select, compare_epsilon
        
    def get_allocation(self, state):
        sub_train_datasetidentifier_2_significance = state["current_sub_train_datasetidentifier_2_significance"]
        sub_train_datasetidentifier_2_epsilon_remain = state["current_sub_train_datasetidentifier_2_epsilon_remain"]
        sub_train_datasetidentifier_2_epsilon_capcity = state["current_sub_train_datasetidentifier_2_epsilon_capcity"]
        target_epsilon_require = state["target_epsilon_require"]
        target_datablock_select_num = state["target_datablock_select_num"]
        job_priority_weight = state["job_priority_weight"]

        temp_datasetidentifier_2_epsilon_z = {
            datasetidentifier: 1.0-sub_train_datasetidentifier_2_epsilon_remain[datasetidentifier]/sub_train_datasetidentifier_2_epsilon_capcity[datasetidentifier]
            for datasetidentifier in sub_train_datasetidentifier_2_epsilon_remain
        }
        count = 0
        selected_datablock_identifiers = []
        calcu_compare_epsilon = 0.0
        
        while count < target_datablock_select_num and len(temp_datasetidentifier_2_epsilon_z.keys()) > 0:
            # 获取随机一个数据集
            datasetidentifier = random.choice(list(temp_datasetidentifier_2_epsilon_z.keys()))
            datablock_epsilon_capacity = sub_train_datasetidentifier_2_epsilon_capcity[datasetidentifier]
            datablock_z = temp_datasetidentifier_2_epsilon_z[datasetidentifier]
            significance_plus_weight = job_priority_weight * sub_train_datasetidentifier_2_significance[datasetidentifier]
            
            is_select, compare_epsilon = self.filter_by_threshold(datablock_epsilon_capacity, datablock_z, target_epsilon_require, significance_plus_weight)
            if is_select:
                count += 1
                selected_datablock_identifiers.append(datasetidentifier)
                calcu_compare_epsilon += compare_epsilon
            del temp_datasetidentifier_2_epsilon_z[datasetidentifier]
        
        return selected_datablock_identifiers, calcu_compare_epsilon