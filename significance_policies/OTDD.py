from significance_policies.BaseSigPolicy import SigPolicy
import json

class OTDDPolicy(SigPolicy):
    def __init__(self):
        super().__init__()
        self._name = "OTDDPolicy"

        self.significance_trace_path = "/home/netlab/DL_lab/opacus_simulation/traces/significance_{}.json".format(self.name)
        with open(self.significance_trace_path, "r+") as f:
            self.significance_trace = json.load(f)

    def get_job_datablock_signficance(self, signficance_state):
        target_dataset_name = signficance_state["target_dataset_name"]
        train_type = signficance_state["train_type"]
        test_type = signficance_state["test_type"]
        return self.significance_trace[target_dataset_name]["sub_train_{}".format(train_type)]["sub_test_{}".format(test_type)]