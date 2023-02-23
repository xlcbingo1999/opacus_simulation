#!/bin/bash
cd /home/netlab/DL_lab/opacus_simulation
beta_test_list=(0.01 0.05 0.1 0.15 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
policy_str="HISPolicy:PBGPolicy:SagePolicy"
logging_date="20230223_HIS_all_test"
prifix="log_schedule_"
if [ ! -d "$prifix$logging_date" ];then
mkdir $prifix$logging_date
echo "创建文件夹成功"
else
echo "文件夹已经存在"
fi
python scheduler_dim_datablock.py --his_betas ${beta_test_list[@]} --policies ${policy_str} --logging_date ${logging_date}
#   --his_gammas ${gamma_test_list[@]} \
#   --his_deltas ${delta_test_list[@]} \
# for beta in ${beta_test_list[@]}
# do
#     for gamma in ${gamma_test_list[@]}
#     do
#         for delta in ${delta_test_list[@]}
#         do
#             python scheduler_dim_datablock.py --his_betas=$beta --his_gammas=$gamma --his_deltas=$delta --policies=$policy_str --logging_date=$logging_date
#         done
#     done
# done
# final_policy_str="PBGPolicy:SagePolicy"
# python scheduler_dim_datablock.py --policies=$final_policy_str
