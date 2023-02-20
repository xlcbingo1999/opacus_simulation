import json
import re

def parse_job_type_tuple(job_type):
    match = re.match('\(\'(.*)\', (\d+)\)', job_type)
    if match is None:
        return None
    model = match.group(1)
    scale_factor = int(match.group(2))
    return (model, scale_factor)

def read_all_throughputs_json_v2(file_name):
    with open(file_name, 'r') as f:
        raw_throughputs = json.load(f)
    parsed_throughputs = {}
    for worker_type in raw_throughputs:
        parsed_throughputs[worker_type] = {}
        for job_type in raw_throughputs[worker_type]:
            key = parse_job_type_tuple(job_type)
            assert(key is not None)
            parsed_throughputs[worker_type][key] = {}
            for other_job_type in raw_throughputs[worker_type][job_type]:
                if other_job_type == 'null':
                    other_key = other_job_type
                else:
                    other_key = parse_job_type_tuple(other_job_type)
                    assert(other_key is not None)
                parsed_throughputs[worker_type][key][other_key] =\
                    raw_throughputs[worker_type][job_type][other_job_type]
    return parsed_throughputs