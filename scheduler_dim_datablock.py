from utils.global_functions import JOB_STATUS_KEY, DATASET_STATUS_KEY, JOB_STATUS_UPDATE_PATH, EVENT_KEY, add_2_map, sort_scores
from utils.global_varible import GLOBAL_PATH, LOGGING_DATE
from utils.get_profiler_significance import get_profiler_selection_result
from utils.logging_tools import get_logger
from utils.throughput_reader import read_all_throughputs_json_v2
from utils.job_generator import generate_all_jobs
from utils.dataset_generator import generate_all_subtrain_datablocks

from policies.PBG import PBGPolicy
from policies.Sage import SagePolicy

import time
from queue import PriorityQueue

import numpy as np
import random
import os


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
    def __init__(self, logger_path, oracle_throughput_path, seed=0):
        self.global_time = 0
        self.queue = PriorityQueue()
        self.all_finished = False
        
        '''
        self.gputype_2_gpu_status = {}
        self.gputype_2_gpu_number = {}
        self.gputype_2_gpu_metadata = {}
        '''
        
        self.sub_train_datasetidentifier_2_dataset_status = {} # 这里必须是一个可以伸缩的map
        self.sub_train_datasetidentifier_2_dataset_metadata = {}
        self.sub_train_datasetidentifier_2_epsilon_capacity = {}
        self.sub_train_datasetidentifier_2_epsilon_remain = {}
        self.sub_train_datasetidentifier_2_submited_time = {}
        self.sub_train_datasetidentifier_2_exhausted_time = {}
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

        '''
        self.jobid_2_gputarget = {}
        self.jobid_2_target_gpu_number = {}
        '''

        self.jobid_2_submited_time = {}
        self.jobid_2_started_time = {}
        self.jobid_2_finished_time = {}

        self.logger = get_logger(logger_path, enable_multiprocess=True)
        self.oracle_throughputs = read_all_throughputs_json_v2(oracle_throughput_path)

        self.initialize_seeds(seed)
        
    def initialize_seeds(self, seed):
        np.random.seed(seed)
        random.seed(seed+1)

        self.job_generator = random.Random()
        self.job_generator.seed(seed+2)
        
    def sched_info(self, msg):
        self.logger.info("TIME[{}] {}".format(self.global_time, msg))
        
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

        '''
        self.jobid_2_gputarget = {}
        self.jobid_2_target_gpu_number = {}
        '''

        self.jobid_2_submited_time = {}
        self.jobid_2_started_time = {}
        self.jobid_2_finished_time = {} 
        self.sched_info("success clear all jobs")

    def clear_all_datasets(self):
        self.sub_train_datasetidentifier_2_dataset_status = {}
        self.sub_train_datasetidentifier_2_dataset_metadata = {}
        self.sub_train_datasetidentifier_2_epsilon_capacity = {}
        self.sub_train_datasetidentifier_2_epsilon_remain = {}
        self.sub_train_datasetidentifier_2_submited_time = {}
        self.sub_train_datasetidentifier_2_exhausted_time = {}
        self.test_datasetname_2_metadata = {}
        self.sched_info("success clear all datasets")

    '''
    def clear_all_gpus(self):
        self.gputype_2_gpu_status = {}
        self.gputype_2_gpu_number = {}
        self.gputype_2_gpu_metadata = {}
    '''


    def update_dataset(self, init_subtrain_datasets_map, init_test_datasets_map):
        # TODO(xlc): 暂时不考虑提交时间的场景, 这里写的比较累赘, 但先照这样写吧
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
            dataset_identifier_2_capacity_map = dispatch_datasetidentifier_2_epsilon_capacity[init_dataset_name] # 这里会得到一个map
            if init_dataset_name not in self.sub_train_datasetidentifier_2_dataset_status:
                self.sub_train_datasetidentifier_2_dataset_status[init_dataset_name] = {}
                self.sub_train_datasetidentifier_2_dataset_metadata[init_dataset_name] = {}
                self.sub_train_datasetidentifier_2_epsilon_capacity[init_dataset_name] = {}
                self.sub_train_datasetidentifier_2_epsilon_remain[init_dataset_name] = {}
                self.sub_train_datasetidentifier_2_submited_time[init_dataset_name] = {}
                self.sub_train_datasetidentifier_2_exhausted_time[init_dataset_name] = {}
            
            for identifier in dataset_identifier_2_capacity_map:
                if identifier not in init_subtrain_datasets_map[init_dataset_name]:
                    self.sched_info("[warning] {} not in dataset config!".format(identifier))
                    continue
                if identifier in self.sub_train_datasetidentifier_2_dataset_status[init_dataset_name]:
                    self.sched_info("[warning] {} already in dataset config!".format(identifier))
                    continue
                self.sub_train_datasetidentifier_2_dataset_status[init_dataset_name][identifier] = DATASET_STATUS_KEY.SUBMITED
                self.sub_train_datasetidentifier_2_dataset_metadata[init_dataset_name][identifier] = init_subtrain_datasets_map[init_dataset_name][identifier]
                self.sub_train_datasetidentifier_2_epsilon_capacity[init_dataset_name][identifier] = dataset_identifier_2_capacity_map[identifier]
                self.sub_train_datasetidentifier_2_epsilon_remain[init_dataset_name][identifier] = dataset_identifier_2_capacity_map[identifier]
                self.sub_train_datasetidentifier_2_submited_time[init_dataset_name][identifier] = self.global_time
                self.sched_info("sucess update dataset [{}-{}]".format(init_dataset_name, identifier))

    '''
    def update_gpu(self, init_gputype_2_num, init_gputype_2_metadata):
        for gpu_type in init_gputype_2_num:
            self.gputype_2_gpu_number[gpu_type] = init_gputype_2_num[gpu_type]
            self.gputype_2_gpu_metadata[gpu_type] = init_gputype_2_metadata[gpu_type]
            self.gputype_2_gpu_status[gpu_type] = True # TODO(xlc): 没有将GPU设置为False的操作
    '''

    def update_jobs(self, jobs_detail_map): # 每次可以增加一批任务
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
                self.jobid_2_datasettargetconfig[id] = {
                    "dataset_name": origin_info["dataset_name"],
                }
                self.jobid_2_trainconfig[id] = {}
                self.jobid_2_target_epsilon[id] = origin_info["EPSILON"]
                self.jobid_2_real_epsilon[id] = 0
                self.jobid_2_submited_time[id] = origin_info["time"]

                # self.jobid_2_target_gpu_number[id] = origin_info["worker_select_num"]
                self.queue.put(SchedEvent(origin_info["time"], EVENT_KEY.JOB_SUBMIT, {"job_id": id}))
                self.sched_info("success add new job {}".format(id))

    def update_max_time(self, max_time):
        self.queue.put(SchedEvent(max_time, EVENT_KEY.MAX_TIME, {}))

    def sche_reflash_job_status(self, job_id, origin_status, new_status):
        self.jobid_2_status[job_id] = new_status
        self.status_2_jobid[origin_status].remove(job_id)
        self.status_2_jobid[new_status].append(job_id)

    def worker_finished_job_callback(self, job_id):
        self.sched_info("Scheduler: Job {job_id} Finished".format(job_id=job_id))
        result = "[Warning] Waiting for setting"
        self.sche_reflash_job_status(job_id, JOB_STATUS_KEY.RUNNING, JOB_STATUS_KEY.FINISHED)
        self.jobid_2_finished_time[job_id] = self.global_time
        self.jobid_2_results[job_id] = result
        self.jobid_2_real_epsilon[job_id] = self.jobid_2_target_epsilon[job_id]
        remain_epsilon = self.jobid_2_target_epsilon[job_id] - self.jobid_2_real_epsilon[job_id]
        dataset_name = self.jobid_2_datasettargetconfig[job_id]["dataset_name"]
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
        self.sched_info("self.jobid_2_status: {}".format(self.jobid_2_status))
        self.sched_info("self.status_2_jobid: {}".format(self.status_2_jobid))
        # self.sched_info("self.jobid_2_gputarget: {}".format(self.jobid_2_gputarget))
        self.sched_info("self.sub_train_datasetidentifier_2_dataset_status")
        for datasetidentifier in self.sub_train_datasetidentifier_2_dataset_status:
            self.sched_info("datasetidentifier[{}]: {}".format(datasetidentifier, self.sub_train_datasetidentifier_2_dataset_status[datasetidentifier]))
        self.sched_info("self.sub_train_datasetidentifier_2_dataset_metadata")
        for datasetidentifier in self.sub_train_datasetidentifier_2_dataset_metadata:
            self.sched_info("datasetidentifier[{}]: {}".format(datasetidentifier, self.sub_train_datasetidentifier_2_dataset_metadata[datasetidentifier]))
        self.sched_info("self.sub_train_datasetidentifier_2_epsilon_capacity")
        for datasetidentifier in self.sub_train_datasetidentifier_2_epsilon_capacity:
            self.sched_info("datasetidentifier[{}]: {}".format(datasetidentifier, self.sub_train_datasetidentifier_2_epsilon_capacity[datasetidentifier]))
        self.sched_info("self.sub_train_datasetidentifier_2_epsilon_remain")
        for datasetidentifier in self.sub_train_datasetidentifier_2_epsilon_remain:
            self.sched_info("datasetidentifier[{}]: {}".format(datasetidentifier, self.sub_train_datasetidentifier_2_epsilon_remain[datasetidentifier]))
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
    
    def get_scheduling_datablock_result(self, policy, target_select_dataset_name, target_epsilon_require, target_datablock_select_num):        
        selected_datablock_identifiers = []
        if target_select_dataset_name not in self.sub_train_datasetidentifier_2_dataset_status:
            return selected_datablock_identifiers
        # train_all_label_distribution = {}
        sub_train_datasetidentifier_2_significance = {}
        sub_train_datasetidentifier_2_epsilon_remain = {}
        sub_train_datasetidentifier_2_epsilon_capcity = {}
        for datablock_identifier in self.sub_train_datasetidentifier_2_dataset_status[target_select_dataset_name].keys():
            # train_all_label_distribution = add_2_map(self.sub_train_datasetidentifier_2_dataset_metadata[target_select_dataset_name][datablock_identifier]["label_distribution"], train_all_label_distribution)
            if self.sub_train_datasetidentifier_2_dataset_status[target_select_dataset_name][datablock_identifier] == DATASET_STATUS_KEY.SUBMITED and self.sub_train_datasetidentifier_2_epsilon_remain[target_select_dataset_name][datablock_identifier] > target_epsilon_require:
                sub_train_datasetidentifier_2_significance[datablock_identifier] = self.sub_train_datasetidentifier_2_dataset_metadata[target_select_dataset_name][datablock_identifier]["significance"] 
                sub_train_datasetidentifier_2_epsilon_remain[datablock_identifier] = self.sub_train_datasetidentifier_2_epsilon_remain[target_select_dataset_name][datablock_identifier]
                sub_train_datasetidentifier_2_epsilon_capcity[datablock_identifier] = self.sub_train_datasetidentifier_2_epsilon_capacity[target_select_dataset_name][datablock_identifier]
        # 在这里接入算法?
        selected_datablock_identifiers = policy.get_allocation(
            sub_train_datasetidentifier_2_significance, 
            sub_train_datasetidentifier_2_epsilon_remain, 
            sub_train_datasetidentifier_2_epsilon_capcity, 
            target_epsilon_require,
            target_datablock_select_num
        )
        # not_selected_datablock_identifiers = [tu[0] for tu in sub_train_sort[target_datablock_select_num:]]
        return selected_datablock_identifiers     

    def sched_for_one_job(self, job_id, policy):
        need_failed_job = False
        dataset_sched_success = False
        if (not need_failed_job) and job_id in self.status_2_jobid[JOB_STATUS_KEY.NO_SCHE]:
            dataset_name = self.jobid_2_datasettargetconfig[job_id]["dataset_name"]
            target_epsilon_require = self.jobid_2_target_epsilon[job_id]
            target_datablock_select_num = self.jobid_2_origininfo[job_id]["datablock_select_num"]
            
            # 需要使用复杂一点的调度策略了
            selected_datablock_identifiers = \
                 self.get_scheduling_datablock_result(policy, dataset_name, target_epsilon_require, target_datablock_select_num)
            if len(selected_datablock_identifiers) > 0:
                self.sched_info("Job [{}] selected datablock identifiers: {}".format(job_id, selected_datablock_identifiers))
                self.jobid_2_datasettargetconfig[job_id]["selected_datablock_identifiers"] = selected_datablock_identifiers
                consume_epsilon = self.jobid_2_origininfo[job_id]["EPSILON"]
                
                for identifier in selected_datablock_identifiers:
                    self.sub_train_datasetidentifier_2_epsilon_remain[dataset_name][identifier] -= consume_epsilon
                    if self.sub_train_datasetidentifier_2_epsilon_remain[dataset_name][identifier] == 0.0:
                        self.sub_train_datasetidentifier_2_dataset_status[dataset_name][identifier] = DATASET_STATUS_KEY.EXHAUST
                        self.sub_train_datasetidentifier_2_exhausted_time[dataset_name][identifier] = self.global_time
                    elif self.sub_train_datasetidentifier_2_epsilon_remain[dataset_name][identifier] < 0.0:
                        raise ValueError("self.sub_train_datasetidentifier_2_epsilon_remain[dataset_name][identifier] < 0.0")

                status_update_path, target_status = self.get_target_job_status_update_path_and_status(job_id, "dataset")
                origin_status, target_status = self.get_job_status_update_origin_target(status_update_path)
                self.sche_reflash_job_status(job_id, origin_status, target_status)
                dataset_sched_success = True
        if not dataset_sched_success:
            need_failed_job = True

        '''
        worker_sched_success = False
        if (not need_failed_job) and job_id in (self.status_2_jobid[JOB_STATUS_KEY.NO_SCHE] + self.status_2_jobid[JOB_STATUS_KEY.DONE_DATASET_SCHED]):
            all_workers_list = list(self.gputype_2_gpu_status.keys())
            if len(all_workers_list) > 0:
                target_worker_id = job_id % len(all_workers_list) # TODO(xlc): 这里暂时不需要改成多卡, 只需要确定目标即可
                target_worker_identifier = all_workers_list[target_worker_id]
                
                if self.gputype_2_gpu_status[target_worker_identifier]: # 判断GPU的状态
                    self.jobid_2_gputarget[job_id] = target_worker_identifier
                    self.gputype_2_gpu_number[target_worker_identifier] -= self.jobid_2_target_gpu_number[job_id]
                    self.sched_info("Job [{}] target_worker_identifier: {} with number: {}".format(job_id, target_worker_identifier, self.jobid_2_target_gpu_number[job_id]))
                    if self.gputype_2_gpu_number[target_worker_identifier] == 0:
                        self.gputype_2_gpu_status[target_worker_identifier] = False
                    elif self.gputype_2_gpu_number[target_worker_identifier] < 0:
                        raise ValueError("self.gputype_2_gpu_number[target_worker_identifier] < 0")
                    status_update_path, target_status = self.get_target_job_status_update_path_and_status(job_id, "gpu")
                    origin_status, target_status = self.get_job_status_update_origin_target(status_update_path)
                    self.sche_reflash_job_status(job_id, origin_status, target_status)
                    worker_sched_success = True

        if not worker_sched_success:
            need_failed_job = True
        '''

        if (not need_failed_job) and job_id in self.status_2_jobid[JOB_STATUS_KEY.DONE_ALL_SCHED]:
            origin_info = self.jobid_2_origininfo[job_id]
            # worker_dataset_config = self.jobid_2_datasettargetconfig[job_id]
            # worker_identifier = self.jobid_2_gputarget[job_id]
            throughput_ratio = origin_info["throughput"]["v100"]
            simulate_duration = origin_info["reference_num_steps"] / throughput_ratio
            self.jobid_2_started_time[job_id] = self.global_time
            get_simulate_finished_time = self.global_time + simulate_duration
            self.queue.put(SchedEvent(get_simulate_finished_time, EVENT_KEY.JOB_REMOVE, {"job_id": job_id}))
            self.sche_reflash_job_status(job_id, JOB_STATUS_KEY.DONE_ALL_SCHED, JOB_STATUS_KEY.RUNNING)

        if need_failed_job:
            status_update_path, target_status = self.get_target_job_status_update_path_and_status(job_id, "failed")
            origin_status, target_status = self.get_job_status_update_origin_target(status_update_path)
            self.sche_reflash_job_status(job_id, origin_status, target_status)

    def schd_end(self):
        self.all_finished = True

    def simulate_start(self, policy):
        self.logger.info("POLICY {} START!".format(policy.name))
        next_event = self.queue.get()
        next_time = next_event.priority
        self.global_time = next_time
        while True:
            if self.all_finished:
                break
            # 处理已经完成的任务
            if next_event.event_key == EVENT_KEY.JOB_REMOVE:
                job_id = next_event.metadata["job_id"]
                self.worker_finished_job_callback(job_id)
            elif next_event.event_key == EVENT_KEY.JOB_SUBMIT:
                job_id = next_event.metadata["job_id"]
                self.sche_reflash_job_status(job_id, JOB_STATUS_KEY.NO_SUBMIT, JOB_STATUS_KEY.NO_SCHE)
                # 调度未调度的任务, 暂时先只考虑job_id的调度即可
                self.sched_for_one_job(job_id, policy)
            elif next_event.event_key == EVENT_KEY.MAX_TIME:
                self.logger.info("FINISH caused by MAX_TIME")
                self.report_status("FINISH MAX_TIME")
                self.schd_end()
            # 更新时间
            next_event = self.queue.get()
            next_time = next_event.priority
            self.global_time = next_time
            if self.check_all_finished_or_failed():
                self.logger.info("FINISH caused by CHECK")
                self.report_status("FINISH CHECK")
                self.schd_end()

        # 计算一些时间和结果

if __name__ == '__main__':
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
    init_dataset_name_2_block_num = {
        "Home_and_Kitchen": 10
    }

    prefix_path = GLOBAL_PATH
    date = LOGGING_DATE
    current_time = time.strftime('%m-%d-%H-%M-%S', time.localtime())
    file_log_name = 'schedule-review-%s' % (current_time)
    result_log_dir_path = '%s/log_schedule_%s' % (prefix_path, date)
    logger_path_prefix = '%s/%s' % (result_log_dir_path, file_log_name)
    oracle_throughput_path = '%s/traces/physical_cluster_throughputs_without_unconsolidated.json' % (prefix_path)

    job_num = 500
    lam = 3600.0
    fixed_datablock_select_num = 1
    jobs_map, reference_max_time = generate_all_jobs(job_num, oracle_throughput_path, lam, 
                                                    fixed_datablock_select_num=fixed_datablock_select_num)
    subtrain_datasets_map, test_datasets_map = generate_all_subtrain_datablocks(init_dataset_name_2_block_num)
    
    comparison_cost_epsilon = 0.01
    comparison_z_threshold = 0.9
    L = 0.01
    U = 10.0
    pbg = PBGPolicy(comparison_cost_epsilon, comparison_z_threshold, L, U)
    sage = SagePolicy()
    policies = [pbg, sage]

    os.mkdir(logger_path_prefix)
    for policy in policies:
        logger_path = '%s/%s.log' % (logger_path_prefix, policy.name)
        sched = Scheduler(logger_path, oracle_throughput_path)
        # sched.update_gpu(init_gputype_2_num, init_gputype_2_metadata)
        sched.update_dataset(subtrain_datasets_map, test_datasets_map)
        sched.update_jobs(jobs_map)
        sched.update_max_time(reference_max_time)
        
        
        sched.simulate_start(policy)
        sched.clear_all_jobs()
        sched.clear_all_datasets()

    