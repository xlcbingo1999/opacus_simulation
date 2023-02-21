from policies.BasePolicy import Policy
import random


class SagePolicy(Policy):
    def __init__(self):
        super().__init__()
        self._name = 'SagePolicy'
        
    def get_allocation(self, sub_train_datasetidentifier_2_significance, 
                    sub_train_datasetidentifier_2_epsilon_remain, 
                    sub_train_datasetidentifier_2_epsilon_capcity, 
                    target_epsilon_consume,
                    target_datablock_select_num):
        temp_datasetidentifier_2_epsilon_z = {
            datasetidentifier: sub_train_datasetidentifier_2_epsilon_remain[datasetidentifier]/sub_train_datasetidentifier_2_epsilon_capcity[datasetidentifier]
            for datasetidentifier in sub_train_datasetidentifier_2_epsilon_remain
        }
        # final_datasetidentifier_2_epsilon_z = {
        #     datasetidentifier: sub_train_datasetidentifier_2_epsilon_remain[datasetidentifier]/sub_train_datasetidentifier_2_epsilon_capcity[datasetidentifier]
        #     for datasetidentifier in sub_train_datasetidentifier_2_epsilon_remain
        # }
        count = 0
        selected_datablock_identifiers = []
        while count < target_datablock_select_num and len(temp_datasetidentifier_2_epsilon_z.keys()) > 0:
            # 获取随机一个数据集
            datasetidentifier = random.choice(list(temp_datasetidentifier_2_epsilon_z.keys()))
            datablock_epsilon_capacity = sub_train_datasetidentifier_2_epsilon_capcity[datasetidentifier]
            datablock_z = temp_datasetidentifier_2_epsilon_z[datasetidentifier]            
            new_z = datablock_z - target_epsilon_consume / datablock_epsilon_capacity
            selected_datablock_identifiers.append(datasetidentifier)
            # final_datasetidentifier_2_epsilon_z[datasetidentifier] = new_z
            del temp_datasetidentifier_2_epsilon_z[datasetidentifier]
        return selected_datablock_identifiers