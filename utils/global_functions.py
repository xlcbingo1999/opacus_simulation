from enum import Enum


def normal_counter(origin_counter):
    sum_value = sum(origin_counter.values(), 0)
    new_counter = {}
    for key in origin_counter:
        key = str(key)
        new_counter[key] = origin_counter[key] / sum_value
    return new_counter

def add_2_map(A, B):
    C = {}
    for key in list(set(A) | set(B)):
        if A.get(key) and B.get(key):
            C.update({key: A.get(key) + B.get(key)})
        else:
            C.update({key: A.get(key) or B.get(key)})
    return C

def sort_scores(score_dict):
    ''' returns the n scores from a name:score dict'''
    lot = [(k,v) for k, v in score_dict.items()] #make list of tuple from scores dict
    nl = []
    while len(lot)> 0:
        nl.append(max(lot, key=lambda x: x[1]))
        lot.remove(nl[-1])
    return nl

class FAILED_RESULT_KEY(Enum):
    WORKER_NO_READY = 1
    JOB_FAILED = 2
    JOB_TYPE_ERROR = 3

class JOB_STATUS_KEY(Enum):
    NO_SUBMIT = -1
    NO_SCHE = 0
    DONE_GPU_SCHED = 1
    DONE_DATASET_SCHED = 2
    DONE_ALL_SCHED = 3
    RUNNING = 4
    FINISHED = 5
    FAILED = 6

class JOB_STATUS_UPDATE_PATH(Enum):
    NOSCHED_2_GPUSCHED = 0
    NOSCHED_2_DATASETSCHED = 1
    GPUSCHED_2_ALLSCHED = 2
    DATASETSCHED_2_ALLSCHED = 3
    NOSCHED_2_FAILED = 4
    GPUSCHED_2_FAILED = 5
    DATASETSCHED_2_FAILED = 6
    ALLSCHED_2_FAILED = 7
    NOSCHED_2_ALLSCHED = 8

class DATASET_STATUS_KEY(Enum):
    NO_SUBMIT = 0
    SUBMITED = 1
    EXHAUST = 2

class EVENT_KEY(Enum):
    JOB_SUBMIT = 0
    JOB_REMOVE = 1
    JOB_START = 2
    WORKER_ADD = 3
    WORKER_REMOVE = 4
    DATASET_ADD = 5
    DATASET_REMOVE = 6
    MAX_TIME = 100000