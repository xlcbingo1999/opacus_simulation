
import re

if __name__ == "__main__":
    file_path = "/home/netlab/DL_lab/opacus_simulation/log_schedule_20230223_HIS_all_test"
    file_name = "/schedule-review-02-23-07-05-41.log"
    key_words = [
        "policy name:", "policy args:", "Finished Job num:", "Failed Job num:", "all_significance:"
    ]
    draw_partern_str = r"(?<=\[).+?(?=\])"
    draw_par = re.compile(draw_partern_str)

    with open(file=file_path+file_name) as f:
        for index, line in enumerate(f):
            for keyword in key_words:
                if keyword in line:
                    print(line[:-1])
        f.close()