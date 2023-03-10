from utils.global_functions import JOB_STATUS_KEY, DATASET_STATUS_KEY, JOB_STATUS_UPDATE_PATH, EVENT_KEY, add_2_map, sort_scores
from utils.global_varible import GLOBAL_PATH
from utils.get_profiler_significance import get_profiler_selection_result
from utils.logging_tools import get_logger
from utils.throughput_reader import read_all_throughputs_json_v2, read_all_accuracy_json
from utils.job_generator import generate_all_jobs
from utils.dataset_generator import generate_all_subtrain_datablocks

from policies.PBG import PBGPolicy
from policies.Sage import SagePolicy
from policies.HIS import HISPolicy
from policies.DPF_HIS_event import DPFHISPolicy
from significance_policies.OTDD import OTDDPolicy
from significance_policies.Temp import TempPolicy

import time
from queue import PriorityQueue

import numpy as np
import random
import os
import argparse
import itertools
import copy

def get_df_config():
    parser = argparse.ArgumentParser(
                description='Sweep through lambda values')
    parser.add_argument('--logging_date', type=str, default="20230305")
    parser.add_argument('--job_num', type=int, default=500)
    parser.add_argument('--history_job_num', type=int, default=500)
    parser.add_argument('--train_type_num', type=int, default=10)
    parser.add_argument('--test_type_num', type=int, default=3)
    parser.add_argument('--fixed_datablock_select_num', type=int, default=None)
    parser.add_argument('--policies', type=str, default="SagePolicy") # PBGPolicy:SagePolicy:DPFHISPolicy HISPolicy:DPFHISPolicy:PBGPolicy:
    parser.add_argument('--significance_policy', type=str, default="OTDDPolicy") # OTDDPolicy
    parser.add_argument('--pbg_comparison_cost_epsilons', type=float, nargs="+", default=[0.0])
    parser.add_argument('--pbg_comparison_z_thresholds', type=float, nargs="+", default=[0.7])
    parser.add_argument('--pbg_Ls', type=float, nargs="+", default=[0.01])
    parser.add_argument('--pbg_Us', type=float, nargs="+", default=[10.0])

    parser.add_argument('--his_betas', type=float, nargs="+", default=[0.01])
    # parser.add_argument('--his_gammas', type=float, nargs="+", default=[0.5])
    # parser.add_argument('--his_deltas', type=float, nargs="+", default=[0.5])
    # parser.add_argument('--his_only_small_flags', type=bool, nargs="+", default=[True])

    parser.add_argument('--dpf_his_betas', type=float, nargs="+", default=[0.01])
    parser.add_argument('--dpf_his_waiting_queue_capacitys', type=int, nargs="+", default=[10])
    
    args = parser.parse_args()
    return args

class SchedEvent(object):
    def __init__(self, priority, event_key, metadata):
        self.priority = priority
        self.event_key = event_key
        self.metadata = metadata
    
    def __lt__(self, other): 
        return self.priority < other.priority
                   
    def __str__(self):
        return '(' + str(self.priority)+',\'' + self.event_key + '\')'

class Scheduler(object):
    def __init__(self, logger, oracle_throughput_path, accuracy_result_path, seed=0):
        self.global_time = 0
        self.queue = PriorityQueue()
        self.all_finished = False
        
        '''
        self.gputype_2_gpu_status = {}
        self.gputype_2_gpu_number = {}
        self.gputype_2_gpu_metadata = {}
        '''
        
        self.sub_train_datasetidentifier_2_dataset_status = {} # ????????????????????????????????????map
        self.sub_train_datasetidentifier_2_dataset_metadata = {}
        self.sub_train_datasetidentifier_2_epsilon_capacity = {}
        self.sub_train_datasetidentifier_2_epsilon_remain = {}
        self.sub_train_datasetidentifier_2_submited_time = {}
        self.sub_train_datasetidentifier_2_exhausted_time = {}
        self.sub_train_datasetidentifier_2_train_type = {}
        self.test_datasetname_2_metadata = {}
        
        self.jobid_2_status = {} # 0: no sche; 1: sched target decide; 2: runnning; 3: success finished; 4: failed;
        self.status_2_jobid = {JOB_STATUS_KEY.NO_SUBMIT: [],
                                JOB_STATUS_KEY.NO_SCHE: [], 
                                # JOB_STATUS_KEY.DONE_GPU_SCHED: [], 
                                # JOB_STATUS_KEY.DONE_DATASET_SCHED: [], 
                                JOB_STATUS_KEY.DONE_ALL_SCHED: [],
                                JOB_STATUS_KEY.RUNNING: [], 
                                JOB_STATUS_KEY.FINISHED: [],
                                JOB_STATUS_KEY.FAILED: []}
        self.jobid_2_results = {}
        self.jobid_2_origininfo = {}

        self.jobid_2_datasettargetconfig = {}
        self.jobid_2_trainconfig = {}

        self.jobid_2_target_epsilon = {}
        self.jobid_2_real_epsilon = {}
        self.jobid_2_priority_weight = {}

        self.jobid_2_submited_time = {}
        self.jobid_2_started_time = {}
        self.jobid_2_finished_time = {}
        self.jobid_2_target_dataset_name = {}
        self.jobid_2_datablock_selected_num = {}
        self.jobid_2_test_type = {}
        self.jobid_2_significance = {}

        self.job_sequence_all_num = 0
        self.current_job_arrival_index = -1
        self.history_job_priority_weights = []
        self.history_job_budget_consumes = []
        self.history_job_target_dataset_name = []
        self.history_job_target_selected_num = []
        self.history_job_test_type = []
        self.history_job_significance = []

        self.logger = logger
        self.oracle_throughputs = read_all_throughputs_json_v2(oracle_throughput_path)
        self.result_accuracy = read_all_accuracy_json(accuracy_result_path)
        self.initialize_seeds(seed)
        
    def initialize_seeds(self, seed):
        np.random.seed(seed)
        random.seed(seed+1)

        self.job_generator = random.Random()
        self.job_generator.seed(seed+2)
        
    def sched_info(self, msg):
        self.logger.info("TIME[{}] {}".format(self.global_time, msg))

    def sched_debug(self, msg):
        self.logger.debug("TIME[{}] {}".format(self.global_time, msg))
        
    def check_all_finished_or_failed(self):
        return (len(self.status_2_jobid[JOB_STATUS_KEY.NO_SUBMIT]) <= 0
            and len(self.status_2_jobid[JOB_STATUS_KEY.NO_SCHE]) <= 0 
            # and len(self.status_2_jobid[JOB_STATUS_KEY.DONE_GPU_SCHED]) <= 0
            # and len(self.status_2_jobid[JOB_STATUS_KEY.DONE_DATASET_SCHED]) <= 0
            and len(self.status_2_jobid[JOB_STATUS_KEY.DONE_ALL_SCHED]) <= 0
            and len(self.status_2_jobid[JOB_STATUS_KEY.RUNNING]) <= 0
        )

    def clear_all_jobs(self):
        self.jobid_2_status = {} # 0: no sche; 1: sched target decide; 2: runnning; 3: success finished; 4: failed;
        self.status_2_jobid = {JOB_STATUS_KEY.NO_SUBMIT: [],
                                JOB_STATUS_KEY.NO_SCHE: [], 
                                # JOB_STATUS_KEY.DONE_GPU_SCHED: [], 
                                # JOB_STATUS_KEY.DONE_DATASET_SCHED: [], 
                                JOB_STATUS_KEY.DONE_ALL_SCHED: [],
                                JOB_STATUS_KEY.RUNNING: [], 
                                JOB_STATUS_KEY.FINISHED: [],
                                JOB_STATUS_KEY.FAILED: []}
        self.jobid_2_results = {}
        self.jobid_2_origininfo = {}
        
        self.jobid_2_datasettargetconfig = {}
        self.jobid_2_trainconfig = {}

        self.jobid_2_target_epsilon = {}
        self.jobid_2_real_epsilon = {}
        self.jobid_2_priority_weight = {}
        self.jobid_2_target_dataset_name = {}
        self.jobid_2_datablock_selected_num = {}
        self.jobid_2_test_type = {}
        self.jobid_2_significance = {}

        self.jobid_2_submited_time = {}
        self.jobid_2_started_time = {}
        self.jobid_2_finished_time = {}

        self.job_sequence_all_num = 0
        self.current_job_arrival_index = -1
        self.history_job_priority_weights = []
        self.history_job_budget_consumes = []
        self.history_job_target_dataset_name = []
        self.history_job_target_selected_num = []
        self.history_job_test_type = []
        self.history_job_significance = []

        self.sched_info("success clear all jobs")

    def clear_all_datasets(self):
        self.sub_train_datasetidentifier_2_dataset_status = {}
        self.sub_train_datasetidentifier_2_dataset_metadata = {}
        self.sub_train_datasetidentifier_2_epsilon_capacity = {}
        self.sub_train_datasetidentifier_2_epsilon_remain = {}
        self.sub_train_datasetidentifier_2_submited_time = {}
        self.sub_train_datasetidentifier_2_exhausted_time = {}
        self.sub_train_datasetidentifier_2_train_type = {}
        self.test_datasetname_2_metadata = {}
        self.sched_info("success clear all datasets")

    '''
    def clear_all_gpus(self):
        self.gputype_2_gpu_status = {}
        self.gputype_2_gpu_number = {}
        self.gputype_2_gpu_metadata = {}
    '''


    def update_dataset(self, init_subtrain_datasets_map, init_test_datasets_map):
        # TODO(xlc): ????????????????????????????????????, ????????????????????????, ?????????????????????
        dispatch_datasetidentifier_2_epsilon_capacity = {}
        for dataset_name in init_subtrain_datasets_map:
            for sub_train_dataset_identifier in init_subtrain_datasets_map[dataset_name]:
                capacity = init_subtrain_datasets_map[dataset_name][sub_train_dataset_identifier]["epsilon_capacity"]
                if dataset_name not in dispatch_datasetidentifier_2_epsilon_capacity:
                    dispatch_datasetidentifier_2_epsilon_capacity[dataset_name] = {}
                dispatch_datasetidentifier_2_epsilon_capacity[dataset_name][sub_train_dataset_identifier] = capacity

        for init_dataset_name in dispatch_datasetidentifier_2_epsilon_capacity:
            if init_dataset_name not in self.test_datasetname_2_metadata:
                self.test_datasetname_2_metadata[init_dataset_name] = init_test_datasets_map[init_dataset_name]
            dataset_identifier_2_capacity_map = dispatch_datasetidentifier_2_epsilon_capacity[init_dataset_name] # ?????????????????????map
            self.logger.info("success add datset[{}] with {} blocks".format(init_dataset_name, len(dataset_identifier_2_capacity_map)))
            if init_dataset_name not in self.sub_train_datasetidentifier_2_dataset_status:
                self.sub_train_datasetidentifier_2_dataset_status[init_dataset_name] = {}
                self.sub_train_datasetidentifier_2_dataset_metadata[init_dataset_name] = {}
                self.sub_train_datasetidentifier_2_epsilon_capacity[init_dataset_name] = {}
                self.sub_train_datasetidentifier_2_epsilon_remain[init_dataset_name] = {}
                self.sub_train_datasetidentifier_2_submited_time[init_dataset_name] = {}
                self.sub_train_datasetidentifier_2_exhausted_time[init_dataset_name] = {}
                self.sub_train_datasetidentifier_2_train_type[init_dataset_name] = {}
            
            for identifier in dataset_identifier_2_capacity_map:
                if identifier not in init_subtrain_datasets_map[init_dataset_name]:
                    self.sched_debug("[warning] {} not in dataset config!".format(identifier))
                    continue
                if identifier in self.sub_train_datasetidentifier_2_dataset_status[init_dataset_name]:
                    self.sched_debug("[warning] {} already in dataset config!".format(identifier))
                    continue
                self.sub_train_datasetidentifier_2_dataset_status[init_dataset_name][identifier] = DATASET_STATUS_KEY.SUBMITED
                self.sub_train_datasetidentifier_2_dataset_metadata[init_dataset_name][identifier] = init_subtrain_datasets_map[init_dataset_name][identifier]
                self.sub_train_datasetidentifier_2_epsilon_capacity[init_dataset_name][identifier] = dataset_identifier_2_capacity_map[identifier]
                self.sub_train_datasetidentifier_2_epsilon_remain[init_dataset_name][identifier] = dataset_identifier_2_capacity_map[identifier]
                self.sub_train_datasetidentifier_2_submited_time[init_dataset_name][identifier] = self.global_time
                self.sub_train_datasetidentifier_2_train_type[init_dataset_name][identifier] = init_subtrain_datasets_map[init_dataset_name][identifier]["train_type"]
                # self.sched_info("sucess update dataset [{}-{}]".format(init_dataset_name, identifier))

    '''
    def update_gpu(self, init_gputype_2_num, init_gputype_2_metadata):
        for gpu_type in init_gputype_2_num:
            self.gputype_2_gpu_number[gpu_type] = init_gputype_2_num[gpu_type]
            self.gputype_2_gpu_metadata[gpu_type] = init_gputype_2_metadata[gpu_type]
            self.gputype_2_gpu_status[gpu_type] = True # TODO(xlc): ?????????GPU?????????False?????????
    '''

    def update_jobs(self, jobs_detail_map, sig_policy): # ??????????????????????????????
        count = 0
        for id in jobs_detail_map:
            origin_info = jobs_detail_map[id]
            if id in self.jobid_2_status:
                self.sched_info("Waring: job {} has existed!".format(id))
                continue
            else:
                self.jobid_2_status[id] = JOB_STATUS_KEY.NO_SUBMIT
                self.status_2_jobid[JOB_STATUS_KEY.NO_SUBMIT].append(id)
                self.jobid_2_results[id] = None
                self.jobid_2_origininfo[id] = origin_info
                # self.jobid_2_gputarget[id] = None
                self.jobid_2_datasettargetconfig[id] = {}
                self.jobid_2_trainconfig[id] = {}
                target_epsilon_consume = origin_info["EPSILON"]
                self.jobid_2_target_epsilon[id] = target_epsilon_consume
                self.jobid_2_real_epsilon[id] = 0
                self.jobid_2_submited_time[id] = origin_info["time"]
                self.jobid_2_priority_weight[id] = origin_info["priority_weight"]
                target_dataset_name = origin_info["dataset_name"]
                self.jobid_2_target_dataset_name[id] = target_dataset_name
                self.jobid_2_datablock_selected_num[id] = origin_info["datablock_select_num"]
                test_type = origin_info["test_type"]
                self.jobid_2_test_type[id] = test_type
                sub_train_datasetidentifier_2_significance = {}
                for datablock_identifier in self.sub_train_datasetidentifier_2_dataset_status[target_dataset_name]:
                    train_type = self.sub_train_datasetidentifier_2_train_type[target_dataset_name][datablock_identifier]
                    signficance_state = self.get_significance_state(sig_policy, target_dataset_name, train_type, test_type, target_epsilon_consume)
                    sub_train_datasetidentifier_2_significance[datablock_identifier] = sig_policy.get_job_datablock_signficance(signficance_state)
                self.jobid_2_significance[id] = sub_train_datasetidentifier_2_significance

                # self.jobid_2_target_gpu_number[id] = origin_info["worker_select_num"]
                self.queue.put(SchedEvent(origin_info["time"], EVENT_KEY.JOB_SUBMIT, {"job_id": id}))
                count += 1
        self.job_sequence_all_num = count
        self.sched_debug("success add new jobs number: {}".format(self.job_sequence_all_num))

    def update_history_jobs(self, history_jobs_map, sig_policy):
        for id in sorted(history_jobs_map):
            self.history_job_priority_weights.append(history_jobs_map[id]["priority_weight"])
            target_epsilon_consume = history_jobs_map[id]["EPSILON"]
            self.history_job_budget_consumes.append(target_epsilon_consume)
            target_dataset_name = history_jobs_map[id]["dataset_name"]
            self.history_job_target_dataset_name.append(target_dataset_name)
            target_selected_num = history_jobs_map[id]["datablock_select_num"]
            self.history_job_target_selected_num.append(target_selected_num)
            test_type = history_jobs_map[id]["test_type"]
            self.history_job_test_type.append(test_type)
            sub_train_datasetidentifier_2_significance = {}
            for datablock_identifier in self.sub_train_datasetidentifier_2_dataset_status[target_dataset_name]:
                train_type = self.sub_train_datasetidentifier_2_train_type[target_dataset_name][datablock_identifier]
                signficance_state = self.get_significance_state(sig_policy, target_dataset_name, train_type, test_type, target_epsilon_consume)
                sub_train_datasetidentifier_2_significance[datablock_identifier] = sig_policy.get_job_datablock_signficance(signficance_state)
            self.history_job_significance.append(sub_train_datasetidentifier_2_significance)
        self.sched_debug("success add new history jobs number: {}".format(len(history_jobs_map)))

    def sche_timely_update_history_job(self, priority_weight, EPSILON, dataset_name, datablock_selected_num, test_type, significance):
        self.history_job_priority_weights.append(priority_weight)
        self.history_job_budget_consumes.append(EPSILON)
        self.history_job_target_dataset_name.append(dataset_name)
        self.history_job_target_selected_num.append(datablock_selected_num)
        self.history_job_test_type.append(test_type)
        self.history_job_significance.append(significance)
        self.sched_debug("success add a new history job")

    def get_job_profiler_accuracy(self, target_dataset_name, train_type, test_type, model_name, epsilon_train):
        # ????????????accuracy?????????, ??????????????????model_name, target_dataset_name, train_type, test_type, epsilon_train 5????????? ???????????????
        return self.result_accuracy[target_dataset_name]["sub_train_{}".format(train_type)]["sub_test_{}".format(test_type)][model_name]["epsilon_consume_{}".format(epsilon_train)]

    def update_max_time(self, max_time):
        self.queue.put(SchedEvent(max_time, EVENT_KEY.MAX_TIME, {}))

    def sche_reflash_job_status(self, job_id, origin_status, new_status):
        self.jobid_2_status[job_id] = new_status
        self.status_2_jobid[origin_status].remove(job_id)
        self.status_2_jobid[new_status].append(job_id)

    def do_submit_job(self, job_id):
        self.current_job_arrival_index += 1
        self.sche_timely_update_history_job(self.jobid_2_priority_weight[job_id], self.jobid_2_target_epsilon[job_id],
                                            self.jobid_2_target_dataset_name[job_id], self.jobid_2_datablock_selected_num[job_id],
                                            self.jobid_2_test_type[job_id], self.jobid_2_significance[job_id])

    def worker_finished_job_callback(self, job_id, policy):
        # self.sched_info("Scheduler: Job {job_id} Finished".format(job_id=job_id))
        dataset_name = self.jobid_2_target_dataset_name[job_id]
        test_type = self.jobid_2_test_type[job_id]
        model_name = self.jobid_2_origininfo[job_id]["model_name"]
        epsilon_consume = self.jobid_2_target_epsilon[job_id]
        selected_datablock_identifiers = self.jobid_2_datasettargetconfig[job_id]["selected_datablock_identifiers"]
        self.jobid_2_results[job_id] = {}
        sign_result = 0.0
        significance_results = []
        accuracy_results = []
        for datablock_identifier in selected_datablock_identifiers:
            significance_results.append(self.jobid_2_significance[job_id][datablock_identifier]) # TODO(xlc): ????????????????????????????????????????????????????????????
            train_type = self.sub_train_datasetidentifier_2_train_type[dataset_name][datablock_identifier]
            accuracy_results.append(self.get_job_profiler_accuracy(dataset_name, train_type, test_type, model_name, epsilon_consume))
        sign_result = np.mean(significance_results)
        self.jobid_2_results[job_id]["significance"] = sign_result            
        acc_result = np.mean(accuracy_results)
        self.jobid_2_results[job_id]["accuracy"] = acc_result

        self.sche_reflash_job_status(job_id, JOB_STATUS_KEY.RUNNING, JOB_STATUS_KEY.FINISHED)
        self.jobid_2_finished_time[job_id] = self.global_time
        self.jobid_2_real_epsilon[job_id] = self.jobid_2_target_epsilon[job_id]
        remain_epsilon = self.jobid_2_target_epsilon[job_id] - self.jobid_2_real_epsilon[job_id]
        dataset_name = self.jobid_2_target_dataset_name[job_id]
        datablock_identifiers = self.jobid_2_datasettargetconfig[job_id]["selected_datablock_identifiers"]
        # worker_identifier = self.jobid_2_gputarget[job_id]
        # worker_consume_num = self.jobid_2_target_gpu_number[job_id]
        
        for identifier in datablock_identifiers:
            self.sub_train_datasetidentifier_2_epsilon_remain[dataset_name][identifier] += remain_epsilon
            if self.sub_train_datasetidentifier_2_epsilon_remain[dataset_name][identifier] > 0.0:
                self.sub_train_datasetidentifier_2_dataset_status[dataset_name][identifier] = DATASET_STATUS_KEY.SUBMITED
        '''
        self.gputype_2_gpu_number[worker_identifier] += worker_consume_num
        if self.gputype_2_gpu_number[worker_identifier] > 0:
            self.gputype_2_gpu_status[worker_identifier] = True
        '''

    def worker_failed_job_callback(self, job_id, failed_result_key):
        self.sched_info("=========  Scheduler: Job Failed! ===========")
        self.sched_info("job_id: {}".format(job_id))
        self.sched_info("failed_result_key: {}".format(failed_result_key))
        self.sched_info("====================")

    def report_status(self, location):
        self.sched_info("======== Scheduler Status in {} ========".format(location))
        self.sched_info("self.status_2_jobid")
        for status in self.status_2_jobid:
            self.sched_info("status_2_jobid[{}]: {}".format(status, self.status_2_jobid[status]))
        # self.sched_info("self.jobid_2_results")
        all_significance = 0.0
        all_accuracy = 0.0
        for job_id in self.jobid_2_results:
            if self.jobid_2_results[job_id] is not None:
                # self.sched_info("jobid_2_results[{}]: {}".format(job_id, self.jobid_2_results[job_id]))
                all_significance += self.jobid_2_results[job_id]["significance"]
                all_accuracy += self.jobid_2_results[job_id]["accuracy"]
        self.sched_info("self.sub_train_datasetidentifier_2_epsilon_remain")
        for datasetname in self.sub_train_datasetidentifier_2_epsilon_remain:
            for datasetidentifier in self.sub_train_datasetidentifier_2_epsilon_remain[datasetname]:
                self.sched_info("sub_train_datasetidentifier_2_epsilon_remain[{}][{}]: {}".format(datasetname, datasetidentifier, self.sub_train_datasetidentifier_2_epsilon_remain[datasetname][datasetidentifier]))
        
        self.sched_info("Finished Job num: {}".format(len(self.status_2_jobid[JOB_STATUS_KEY.FINISHED])))
        self.sched_info("Failed Job num: {}".format(len(self.status_2_jobid[JOB_STATUS_KEY.FAILED])))
        self.sched_info("all_significance: {}".format(all_significance))
        self.sched_info("all_accuracy: {}".format(all_accuracy))
        self.sched_info("==================================")

    def get_target_job_status_update_path_and_status(self, job_id, operator):
        origin_status = self.jobid_2_status[job_id]
        update_path = None
        new_status = None
        if operator == "dataset":
            if origin_status == JOB_STATUS_KEY.NO_SCHE:
                # self.sched_info("in: dataset 1")
                update_path = JOB_STATUS_UPDATE_PATH.NOSCHED_2_ALLSCHED
                new_status = JOB_STATUS_KEY.DONE_ALL_SCHED
        elif operator == "failed":
            if origin_status == JOB_STATUS_KEY.NO_SCHE:
                update_path = JOB_STATUS_UPDATE_PATH.NOSCHED_2_FAILED
                new_status = JOB_STATUS_KEY.FAILED
            elif origin_status == JOB_STATUS_KEY.DONE_ALL_SCHED:
                update_path = JOB_STATUS_UPDATE_PATH.ALLSCHED_2_FAILED
                new_status = JOB_STATUS_KEY.FAILED
        return update_path, new_status

    def get_job_status_update_origin_target(self, status_update_path):
        if status_update_path == JOB_STATUS_UPDATE_PATH.NOSCHED_2_GPUSCHED:
            origin_status = JOB_STATUS_KEY.NO_SCHE
            target_status = JOB_STATUS_KEY.DONE_GPU_SCHED
        elif status_update_path == JOB_STATUS_UPDATE_PATH.NOSCHED_2_DATASETSCHED:
            origin_status = JOB_STATUS_KEY.NO_SCHE
            target_status = JOB_STATUS_KEY.DONE_DATASET_SCHED
        elif status_update_path == JOB_STATUS_UPDATE_PATH.GPUSCHED_2_ALLSCHED:
            origin_status = JOB_STATUS_KEY.DONE_GPU_SCHED
            target_status = JOB_STATUS_KEY.DONE_ALL_SCHED
        elif status_update_path == JOB_STATUS_UPDATE_PATH.DATASETSCHED_2_ALLSCHED:
            origin_status = JOB_STATUS_KEY.DONE_DATASET_SCHED
            target_status = JOB_STATUS_KEY.DONE_ALL_SCHED
        elif status_update_path == JOB_STATUS_UPDATE_PATH.NOSCHED_2_FAILED:
            origin_status = JOB_STATUS_KEY.NO_SCHE
            target_status = JOB_STATUS_KEY.FAILED
        elif status_update_path == JOB_STATUS_UPDATE_PATH.GPUSCHED_2_FAILED:
            origin_status = JOB_STATUS_KEY.DONE_GPU_SCHED
            target_status = JOB_STATUS_KEY.FAILED
        elif status_update_path == JOB_STATUS_UPDATE_PATH.DATASETSCHED_2_FAILED:
            origin_status = JOB_STATUS_KEY.DONE_DATASET_SCHED
            target_status = JOB_STATUS_KEY.FAILED
        elif status_update_path == JOB_STATUS_UPDATE_PATH.ALLSCHED_2_FAILED:
            origin_status = JOB_STATUS_KEY.DONE_ALL_SCHED
            target_status = JOB_STATUS_KEY.FAILED
        elif status_update_path == JOB_STATUS_UPDATE_PATH.NOSCHED_2_ALLSCHED:
            origin_status = JOB_STATUS_KEY.NO_SCHE
            target_status = JOB_STATUS_KEY.DONE_ALL_SCHED
        return origin_status, target_status
    
    def get_runtime_state(self, policy, job_id_2_dataset_name, job_id_2_target_epsilon_require, job_id_2_target_datablock_selected_num, job_id_2_job_priority_weight, job_id_2_test_type, job_id_2_significance):
        state = {}
        state["job_id_2_target_dataset_name"] = job_id_2_dataset_name
        state["job_id_2_target_epsilon_require"] = job_id_2_target_epsilon_require
        state["job_id_2_target_datablock_selected_num"] = job_id_2_target_datablock_selected_num
        state["job_id_2_job_priority_weight"] = job_id_2_job_priority_weight
        state["job_id_2_test_type"] = job_id_2_test_type
        state["job_id_2_significance"] = job_id_2_significance

        state["current_sub_train_datasetidentifier_2_epsilon_remain"] = copy.deepcopy(self.sub_train_datasetidentifier_2_epsilon_remain)
        state["current_sub_train_datasetidentifier_2_epsilon_capcity"] = copy.deepcopy(self.sub_train_datasetidentifier_2_epsilon_capacity)

        if policy.name == "HISPolicy" or policy.name == "DPFHISPolicy":
            state["job_arrival_index"] = self.current_job_arrival_index
            state["all_job_sequence_num"] = self.job_sequence_all_num
            state["history_job_priority_weights"] = self.history_job_priority_weights
            state["history_job_budget_consumes"] = self.history_job_budget_consumes
            state["history_job_target_dataset_name"] = self.history_job_target_dataset_name
            state["history_job_target_datablock_selected_num"] = self.history_job_target_selected_num
            state["history_job_significance"] = self.history_job_significance
 
        return state

    def get_significance_state(self, policy, target_dataset_name, train_type, test_type, target_epsilon_consume):
        signficance_state = {}
        signficance_state["target_dataset_name"] = target_dataset_name
        signficance_state["train_type"] = train_type
        signficance_state["test_type"] = test_type
        if policy.name == "TempPolicy":
            signficance_state["epsilon_consume"] = target_epsilon_consume
        return signficance_state

    def get_scheduling_datablock_result(self, policy, job_id_2_dataset_name, job_id_2_target_epsilon_require, job_id_2_target_datablock_selected_num, job_id_2_job_priority_weight, job_id_2_test_type, job_id_2_significance):        
        job_2_selected_datablock_identifiers = []
        # ??????????????????????
        state = self.get_runtime_state(policy, job_id_2_dataset_name, job_id_2_target_epsilon_require, job_id_2_target_datablock_selected_num, job_id_2_job_priority_weight, job_id_2_test_type, job_id_2_significance)
        job_2_selected_datablock_identifiers, calcu_compare_epsilon = policy.get_allocation(state)
        # not_selected_datablock_identifiers = [tu[0] for tu in sub_train_sort[target_datablock_select_num:]]
        return job_2_selected_datablock_identifiers, calcu_compare_epsilon

    def sched_for_no_sched_jobs(self, policy):
        
        if self.current_job_arrival_index + 1 < self.job_sequence_all_num and len(self.status_2_jobid[JOB_STATUS_KEY.NO_SCHE]) < policy.waiting_queue_capacity:
            self.logger.debug("No for schedule time casue: [len(self.status_2_jobid[JOB_STATUS_KEY.NO_SCHE]): {}]".format(len(self.status_2_jobid[JOB_STATUS_KEY.NO_SCHE])))
            return
        job_id_2_dataset_name = {job_id: self.jobid_2_target_dataset_name[job_id] for job_id in self.status_2_jobid[JOB_STATUS_KEY.NO_SCHE]}
        job_id_2_target_epsilon_require = {job_id: self.jobid_2_target_epsilon[job_id] for job_id in self.status_2_jobid[JOB_STATUS_KEY.NO_SCHE]}
        job_id_2_target_datablock_selected_num = {job_id: self.jobid_2_datablock_selected_num[job_id] for job_id in self.status_2_jobid[JOB_STATUS_KEY.NO_SCHE]}
        job_id_2_job_priority_weight = {job_id: self.jobid_2_priority_weight[job_id] for job_id in self.status_2_jobid[JOB_STATUS_KEY.NO_SCHE]}
        job_id_2_test_type = {job_id: self.jobid_2_test_type[job_id] for job_id in self.status_2_jobid[JOB_STATUS_KEY.NO_SCHE]}
        job_id_2_significance = {job_id: self.jobid_2_significance[job_id] for job_id in self.status_2_jobid[JOB_STATUS_KEY.NO_SCHE]}

        job_2_selected_datablock_identifiers, calcu_compare_epsilon = \
            self.get_scheduling_datablock_result(policy, job_id_2_dataset_name, job_id_2_target_epsilon_require, job_id_2_target_datablock_selected_num, job_id_2_job_priority_weight, job_id_2_test_type, job_id_2_significance)
        success_sched_job_ids = set()
        if len(job_2_selected_datablock_identifiers) > 0:
            self.sched_info("Jobs selected datablock identifiers: {}".format(job_2_selected_datablock_identifiers))
            for temp_job_id, identifier in job_2_selected_datablock_identifiers:
                if "selected_datablock_identifiers" not in self.jobid_2_datasettargetconfig[temp_job_id]:
                    self.jobid_2_datasettargetconfig[temp_job_id]["selected_datablock_identifiers"] = []
                consume_epsilon = self.jobid_2_origininfo[temp_job_id]["EPSILON"]
                dataset_name = job_id_2_dataset_name[temp_job_id]
                if self.sub_train_datasetidentifier_2_epsilon_remain[dataset_name][identifier] >= consume_epsilon:
                    self.sub_train_datasetidentifier_2_epsilon_remain[dataset_name][identifier] -= consume_epsilon # calcu_compare_epsilon
                    self.jobid_2_datasettargetconfig[temp_job_id]["selected_datablock_identifiers"].append(identifier)
                    success_sched_job_ids.add(temp_job_id)
                if self.sub_train_datasetidentifier_2_epsilon_remain[dataset_name][identifier] <= 0.0:
                    self.sub_train_datasetidentifier_2_dataset_status[dataset_name][identifier] = DATASET_STATUS_KEY.EXHAUST
                    self.sub_train_datasetidentifier_2_exhausted_time[dataset_name][identifier] = self.global_time

        need_failed_job = copy.deepcopy(self.status_2_jobid[JOB_STATUS_KEY.NO_SCHE])
        for temp_job_id in success_sched_job_ids:
            status_update_path, target_status = self.get_target_job_status_update_path_and_status(temp_job_id, "dataset")
            origin_status, target_status = self.get_job_status_update_origin_target(status_update_path)
            self.sche_reflash_job_status(temp_job_id, origin_status, target_status)
            need_failed_job.remove(temp_job_id)
                

        for temp_job_id in self.status_2_jobid[JOB_STATUS_KEY.DONE_ALL_SCHED]:
            origin_info = self.jobid_2_origininfo[temp_job_id]
            throughput_ratio = origin_info["throughput"]["v100"]
            simulate_duration = origin_info["reference_num_steps"] / throughput_ratio
            self.jobid_2_started_time[temp_job_id] = self.global_time
            get_simulate_finished_time = self.global_time + simulate_duration
            self.queue.put(SchedEvent(get_simulate_finished_time, EVENT_KEY.JOB_REMOVE, {"job_id": temp_job_id}))
            self.sche_reflash_job_status(temp_job_id, JOB_STATUS_KEY.DONE_ALL_SCHED, JOB_STATUS_KEY.RUNNING)

        for temp_job_id in need_failed_job:
            self.sched_debug("failed job [{}]".format(temp_job_id))
            status_update_path, target_status = self.get_target_job_status_update_path_and_status(temp_job_id, "failed")
            origin_status, target_status = self.get_job_status_update_origin_target(status_update_path)
            self.sche_reflash_job_status(temp_job_id, origin_status, target_status)

    def schd_end(self):
        self.all_finished = True

    def simulate_start(self, policy):
        self.logger.info("POLICY {} START!".format(policy.name))
        policy.report_state()
        next_event = self.queue.get()
        next_time = next_event.priority
        self.global_time = next_time
        while True:
            if self.all_finished:
                break
            # ???????????????????????????
            if next_event.event_key == EVENT_KEY.JOB_REMOVE:
                job_id = next_event.metadata["job_id"]
                self.worker_finished_job_callback(job_id, policy)
            elif next_event.event_key == EVENT_KEY.JOB_SUBMIT:
                job_id = next_event.metadata["job_id"]
                self.do_submit_job(job_id)
                self.sche_reflash_job_status(job_id, JOB_STATUS_KEY.NO_SUBMIT, JOB_STATUS_KEY.NO_SCHE)
                # ????????????????????????, ??????????????????job_id???????????????
                self.sched_for_no_sched_jobs(policy)
            elif next_event.event_key == EVENT_KEY.MAX_TIME:
                self.logger.info("FINISH caused by MAX_TIME")
                self.report_status("FINISH MAX_TIME")
                self.schd_end()
            # ????????????
            if self.check_all_finished_or_failed():
                self.logger.info("FINISH caused by CHECK")
                self.report_status("FINISH CHECK")
                self.schd_end()
            else:
                if not self.queue.empty():
                    next_event = self.queue.get()
                    next_time = next_event.priority
                    self.global_time = next_time

        # ???????????????????????????

def do_one_game(logger, oracle_throughput_path, accuracy_result_path,
                subtrain_datasets_map, test_datasets_map,
                jobs_map, history_jobs_map, reference_max_time,
                policy_item, sig_policy_item):
    sched = Scheduler(logger, oracle_throughput_path, accuracy_result_path)
    sched.update_dataset(subtrain_datasets_map, test_datasets_map)
    sched.update_jobs(jobs_map, sig_policy_item)
    sched.update_history_jobs(history_jobs_map, sig_policy_item)
    sched.update_max_time(reference_max_time)
    sched.simulate_start(policy_item)
    sched.clear_all_jobs()
    sched.clear_all_datasets()
    del sched

if __name__ == '__main__':
    args = get_df_config()
    init_gputype_2_num = {
        'v100': 12,
        'p100': 12,
        'k80': 12
    }
    init_gputype_2_metadata = {
        'v100': None,
        'p100': None,
        'k80': None
    }
    init_dataset_name_2_train_type_num = {
        "EMNIST": args.train_type_num
    }
    init_dataset_name_2_test_type_num = {
        "EMNIST": args.test_type_num
    }

    prefix_path = GLOBAL_PATH
    date = args.logging_date
    current_time = time.strftime('%m-%d-%H-%M-%S', time.localtime())
    file_log_name = 'schedule-review-%s' % (current_time)
    result_log_dir_path = '%s/log_schedule_%s' % (prefix_path, date)
    logger_path_prefix = '%s/%s' % (result_log_dir_path, file_log_name)
    oracle_throughput_path = '%s/traces/physical_cluster_throughputs_without_unconsolidated.json' % (prefix_path)
    accuracy_result_path = '%s/traces/accuracy_result.json' % (prefix_path)

    job_num = args.job_num
    history_num = args.history_job_num
    lam = 3600.0
    fixed_datablock_select_num = args.fixed_datablock_select_num
    jobs_map, history_jobs_map, reference_max_time = generate_all_jobs(job_num, history_num, init_dataset_name_2_test_type_num, oracle_throughput_path, lam, 
                                                    fixed_datablock_select_num=fixed_datablock_select_num)
    subtrain_datasets_map, test_datasets_map = generate_all_subtrain_datablocks(init_dataset_name_2_train_type_num)
    
    logger_path = '%s.log' % (logger_path_prefix)
    logger = get_logger(logger_path, enable_multiprocess=False)
    policies = args.policies.split(":")
    significance_policy_name = args.significance_policy
    
    if significance_policy_name == "OTDDPolicy":
        sig_policy_item = OTDDPolicy()
    elif significance_policy_name == "tempPolicy":
        sig_policy_item = TempPolicy()
    logger.info("SigPOLICY {} START!".format(sig_policy_item.name))

    for policy in policies:
        args_product_list = []
        if policy == "PBGPolicy":
            comparison_cost_epsilon_list = args.pbg_comparison_cost_epsilons
            comparison_z_threshold_list = args.pbg_comparison_z_thresholds
            L_list = args.pbg_Ls
            U_list = args.pbg_Us
            args_product_list = [d for d in itertools.product(comparison_cost_epsilon_list, comparison_z_threshold_list, L_list, U_list)]
        elif policy == "HISPolicy":
            beta_list = args.his_betas
            # gamma_list = args.his_gammas
            # delta_list = args.his_deltas
            # only_small_flag_list = args.his_only_small_flags
            args_product_list = beta_list
        elif policy == "DPFHISPolicy":
            beta_list = args.dpf_his_betas
            waiting_queue_capacity_list = args.dpf_his_waiting_queue_capacitys
            args_product_list = [d for d in itertools.product(beta_list, waiting_queue_capacity_list)]
        
        if policy == "PBGPolicy":
            for temp_arg in args_product_list:
                comparison_cost_epsilon, comparison_z_threshold, L, U = temp_arg
                policy_item = PBGPolicy(comparison_cost_epsilon, comparison_z_threshold, L, U, logger)
                do_one_game(logger, oracle_throughput_path, accuracy_result_path,
                            subtrain_datasets_map, test_datasets_map,
                            jobs_map, history_jobs_map, reference_max_time,
                            policy_item, sig_policy_item)
        elif policy == "HISPolicy":
            for temp_arg in args_product_list:
                beta = temp_arg
                policy_item = HISPolicy(beta, logger)
                do_one_game(logger, oracle_throughput_path, accuracy_result_path,
                            subtrain_datasets_map, test_datasets_map,
                            jobs_map, history_jobs_map, reference_max_time,
                            policy_item, sig_policy_item)
        elif policy == "SagePolicy":
            policy_item = SagePolicy(logger)
            do_one_game(logger, oracle_throughput_path, accuracy_result_path,
                        subtrain_datasets_map, test_datasets_map,
                        jobs_map, history_jobs_map, reference_max_time,
                        policy_item, sig_policy_item)
        elif policy == "DPFHISPolicy":
            print(policy)
            for temp_arg in args_product_list:
                beta, waiting_queue_capacity = temp_arg
                policy_item = DPFHISPolicy(beta, waiting_queue_capacity, logger)
                do_one_game(logger, oracle_throughput_path, accuracy_result_path,
                            subtrain_datasets_map, test_datasets_map,
                            jobs_map, history_jobs_map, reference_max_time,
                            policy_item, sig_policy_item)