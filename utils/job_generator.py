import random
from utils.job_table import get_job_table
import math
from utils.throughput_reader import read_all_throughputs_json_v2

def _generate_duration(rng):
    # Sample the job duration from the Philly distribution.
    if rng.random() >= 0.8:
        run_time = 60 * (10 ** rng.uniform(3, 4))
    else:
        run_time = 60 * (10 ** rng.uniform(1.5, 3))
    return run_time

def _generate_memory(rng):
    mem = 16
    r = rng.uniform(0, 1)
    if r >= 0.75:
        mem = 32
    elif 0.5 <= r < 0.75:
        mem = 24
    elif 0.25 <= r < 0.5:
        mem = 16
    elif r < 0.25:
        mem = 8
    return mem

def _generate_privacy_budget(rng):
    privacy_epsilon = 1.0
    privacy_delta = 1e-7
    r = rng.uniform(0, 1)
    # if r >= 0.75:
    #     privacy_epsilon = 1.0
    # elif 0.25 <= r < 0.75:
    #     privacy_epsilon = 5.0
    # elif r < 0.25:
    #     privacy_epsilon = 10.0
    if r >= 0.25:
        privacy_epsilon = 1.0
    else:
        privacy_epsilon = 5.0
    return privacy_epsilon, privacy_delta

def _generate_worker_select_num(rng):
    # Sample the scale factor from the Philly distribution.
    select_num = 1
    r = rng.uniform(0, 1)
    if 0.8 <= r <= 0.9:
        select_num = 2
    elif 0.91 <= r <= 0.98:
        select_num = 4
    elif 0.99 <= r:
        select_num = 8
    return select_num

def _generate_datablock_select_num(rng):
    num = 1
    r = rng.uniform(0, 1)
    if 0.80 <= r <= 0.95:
        num = 2
    elif 0.95 <= r <= 0.98:
        num = 4
    elif 0.99 <= r:
        num = 8
    return num

def _generate_SLO(rng):
    SLO = 1.0
    r = rng.uniform(0, 1)
    if 0.0 <= r < 0.33:
        SLO = 1.2
    elif 0.33 <= r < 0.67:
        SLO = 2.0
    else:
        SLO = 10.0
    return SLO

def _generate_test_type(rng, all_test_type_num):
    r = rng.uniform(0, 1)
    split_points = [p/all_test_type_num for p in range(all_test_type_num)] # 4 [0, 0.25, 0.5, 0.75]
    result = 0
    for i in range(all_test_type_num-1):
        if split_points[i] <= r < split_points[i+1]: # [0, 0.25], [0.25, 0.5], [0.5, 0.75]
            result = i
    if r >= split_points[all_test_type_num-1]: # [0.75, ]
        result = all_test_type_num-1
    return result

def _generate_arrival_time_delta(rng, rate_parameter):
    """Samples job interarrival rate from a Poisson distribution according
        to the specified rate parameter."""
    return -math.log(1.0 - rng.random()) / rate_parameter

def generate_job(init_dataset_name_2_test_type_num,
                 throughputs, last_job_arrival_time, lam,
                 reference_worker_type='v100', 
                 fixed_job_duration=None, 
                 fixed_privacy_budget=None,
                 fixed_datablock_select_num=None,

                 generate_multi_gpu_jobs=False,
                 generate_multi_priority_jobs=False,
                 always_generate_worker_select_num=True,

                 arrival_time_delta_generator_func=_generate_arrival_time_delta,
                 duration_generator_func=_generate_duration,
                 SLO_genrator_func=_generate_SLO,
                 privacy_budget_generator_func=_generate_privacy_budget,
                 worker_select_num_generator_func=_generate_worker_select_num,
                 datablock_select_num_generator_func=_generate_datablock_select_num,
                 test_type_generator_func=_generate_test_type
                 ):
    """Generates a new job.

       Args:
         init_dataset_name_2_test_type_num: 
         throughputs: A dict containing pre-measured throughputs.
         reference_worker_type: The worker type to use when calculating steps.
         rng: A random number generator for selecting job parameters.
         job_id: The job's ID.
         fixed_job_duration: If set, fixes the duration to the specified value.
         generate_multi_gpu_jobs: If set, generate a scale factor >= 1.
         generate_multi_priority_jobs: If set, generate a priority >= 1.
         run_dir: The directory to run the job from.
         scale_factor_generator_func: A function that accepts an RNG parameter
                                      and returns a job size.
         duration_generator_func: A function that accepts an RNG parameter and
                                  returns a job duration in seconds.
         scale_factor_rng: A random number generator specifically for
                           generating scale factors.
         duration_rng: A random number generator specifically for generating
                       durations.
         SLO_rng: If set, generate an SLO >= 1 using this RNG.
         always_generate_scale_factor: If set, generate a scale factor
                                       regardless of whether user has
                                       requested multi-GPU jobs.
      Returns:
        The generated Job.
    """
    rng = random.Random()

    if always_generate_worker_select_num:
        worker_select_num = worker_select_num_generator_func(rng)
    else:
        # NOTE: We select the job template here to maintain backwards
        # compatability with scripts/utils/generate_trace.py
        Job_table = get_job_table()
        job_template = rng.choice(Job_table)
        if generate_multi_gpu_jobs and job_template.distributed:
            worker_select_num = worker_select_num_generator_func(rng)
        else:
            worker_select_num = 1
    if not generate_multi_gpu_jobs:
        worker_select_num = 1

    if fixed_datablock_select_num is not None:
        datablock_select_num = fixed_datablock_select_num
    else:
        datablock_select_num = datablock_select_num_generator_func(rng)

    if fixed_job_duration is not None: # FEATURE(xlc): ???????????????????????????v100??????????????????
        run_time = fixed_job_duration
    else:
        run_time = duration_generator_func(rng)

    if fixed_privacy_budget is not None:
        privacy_epsilon, privacy_delta = fixed_privacy_budget
    else:
        privacy_epsilon, privacy_delta = privacy_budget_generator_func(rng)

    assert(run_time > 0)
    assert(worker_select_num >= 1 and worker_select_num <= 8)

    # Sample the job type. # DEBUG(xlc): ?????????scale_factor????????????????????????, ????????????
    while True:
        Job_table = get_job_table()
        job_template = rng.choice(Job_table)
        job_type = job_template.model
        key = (job_type, worker_select_num)
        job_throughput = [throughputs[worker_type][key]['null'] for worker_type in throughputs.keys()]
        if (0.0 not in job_throughput) and (worker_select_num == 1 or
            (worker_select_num > 1 and job_template.distributed)):
            break

    # Compute the number of steps the job will run for given its duration.
    key = (job_type, worker_select_num)
    assert(key in throughputs[reference_worker_type])
    reference_throughput = throughputs[reference_worker_type][key]['null']
    num_steps = run_time * reference_throughput
    job_throughput = {worker_type:throughputs[worker_type][key]['null'] for worker_type in throughputs.keys()}
    assert(num_steps > 0)

    # Optionally assign a priority to the job.
    priority_weight = 1.0
    if generate_multi_priority_jobs:
        r = rng.uniform(0, 1)
        if 0.0 <= r <= 0.2:
            priority_weight = 5.0

    # Optionally assign an SLO to the job.
    SLO = SLO_genrator_func(rng)

    # ??????test dataset
    all_test_type_num = init_dataset_name_2_test_type_num[job_template.target_dataset]
    test_type = test_type_generator_func(rng, all_test_type_num)

    next_job_arrival_time = arrival_time_delta_generator_func(rng, 1.0 / lam) + last_job_arrival_time

    job = {
        "time": next_job_arrival_time,
        "model_name": job_template.model_name,
        "model_detail": job_template.model,
        "dataset_name": job_template.target_dataset,
        "datablock_select_num": datablock_select_num,
        "worker_select_num": worker_select_num,
        "EPSILON": privacy_epsilon,
        "DELTA": privacy_delta,
        "SLO": SLO,
        "priority_weight": priority_weight,
        "reference_num_steps": num_steps,
        "throughput": job_throughput,
        "test_type": test_type
    }

    return job

def generate_all_jobs(test_num, history_num, init_dataset_name_2_test_type_num, throughputs_path, lam, fixed_datablock_select_num=None):
    throughputs = read_all_throughputs_json_v2(throughputs_path)
    
    history_last_job_arrival_time = 0.0
    history_jobs_map = {}
    for index in range(history_num):
        job = generate_job(init_dataset_name_2_test_type_num, throughputs, history_last_job_arrival_time, lam, 
                        fixed_datablock_select_num=fixed_datablock_select_num)
        history_jobs_map[history_num - 1 - index] = job
        history_last_job_arrival_time = job["time"]
    for index in range(history_num):
        history_jobs_map[index]["time"] = history_last_job_arrival_time - history_jobs_map[index]["time"]

    max_real_calculate_time = 0.0
    last_job_arrival_time = 0.0
    test_jobs_map = {}
    for index in range(test_num):
        job = generate_job(init_dataset_name_2_test_type_num, throughputs, last_job_arrival_time, lam, 
                        fixed_datablock_select_num=fixed_datablock_select_num)
        test_jobs_map[index] = job
        last_job_arrival_time = job["time"]
    
        for worker_type in job["throughput"].keys():
            temp_calculate_time = job["reference_num_steps"] / job["throughput"][worker_type]
            if last_job_arrival_time + temp_calculate_time > max_real_calculate_time:
                max_real_calculate_time = last_job_arrival_time + temp_calculate_time
    reference_max_time = 2 * max_real_calculate_time
    return test_jobs_map, history_jobs_map, reference_max_time