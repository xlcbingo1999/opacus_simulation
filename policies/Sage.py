from policies.BasePolicy import Policy
import random


class SagePolicy(Policy):
    def __init__(self):
        super().__init__()
        self._name = 'SagePolicy'

    def report_state(self, logger):
        logger.info("policy name: {}".format(self._name))
        logger.info("policy args: None")    
    
    def get_allocation(self, state):
        sub_train_datasetidentifier_2_significance = state["current_sub_train_datasetidentifier_2_significance"]
        sub_train_datasetidentifier_2_epsilon_remain = state["current_sub_train_datasetidentifier_2_epsilon_remain"]
        sub_train_datasetidentifier_2_epsilon_capcity = state["current_sub_train_datasetidentifier_2_epsilon_capcity"]
        target_epsilon_require = state["target_epsilon_require"]
        target_datablock_select_num = state["target_datablock_select_num"]
        job_priority_weight = state["job_priority_weight"]
        
        temp_datasetidentifier_2_epsilon_z = {
            datasetidentifier: sub_train_datasetidentifier_2_epsilon_remain[datasetidentifier]/sub_train_datasetidentifier_2_epsilon_capcity[datasetidentifier]
            for datasetidentifier in sub_train_datasetidentifier_2_epsilon_remain
        }
        # final_datasetidentifier_2_epsilon_z = {
        #     datasetidentifier: sub_train_datasetidentifier_2_epsilon_remain[datasetidentifier]/sub_train_datasetidentifier_2_epsilon_capcity[datasetidentifier]
        #     for datasetidentifier in sub_train_datasetidentifier_2_epsilon_remain
        # }
        count = 0
        calcu_compare_epsilon = 0.0
        selected_datablock_identifiers = []
        while count < target_datablock_select_num and len(temp_datasetidentifier_2_epsilon_z.keys()) > 0:
            # 获取随机一个数据集
            datasetidentifier = random.choice(list(temp_datasetidentifier_2_epsilon_z.keys()))
            datablock_epsilon_capacity = sub_train_datasetidentifier_2_epsilon_capcity[datasetidentifier]
            datablock_z = temp_datasetidentifier_2_epsilon_z[datasetidentifier]            
            selected_datablock_identifiers.append(datasetidentifier)
            # final_datasetidentifier_2_epsilon_z[datasetidentifier] = new_z
            del temp_datasetidentifier_2_epsilon_z[datasetidentifier]
            count += 1
        return selected_datablock_identifiers, calcu_compare_epsilon