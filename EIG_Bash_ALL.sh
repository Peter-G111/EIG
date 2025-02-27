#!/bin/bash

# usage: bash EIG_Notebook.jl
# runs `EIG_Notebook.jl` with different parameters

# run the original t_max experiments
echo "running original t_max runs"

ult_timeout=3600

for oracle_id in $(seq 2 -1 1);  # do network oracles first
do
    for grid_size in $(seq 3 10);
    do
        for trial_num in $(seq 10);
        do
            echo "running grid_size=$grid_size trial_num=$trial_num orig t_max with oracle_id=$oracle_id"
            
            # usage: EIG_Notebook.jl <total_timeout for each oracle int> <t_max_offset int> <use_orig_t_max bool> <grid_size> <trial_num> <A_oracle_num> <D_oracle_num>
            julia "EIG_Notebook.jl" $ult_timeout "0" "true" $grid_size $trial_num $oracle_id $oracle_id  # timeout of 1 hour for each run
        done
    done
done

# # usage: bash EIG_Notebook.jl <total_timeout for each oracle> <oracle_num>
# # runs `EIG_Notebook.jl` with different parameters

# # run the original t_max experiments
# echo "running original t_max runs"

# for grid_size in $(seq 3 10);
# do
#     for trial_num in $(seq 10);
#     do
#         echo "running grid_size=$grid_size trial_num=$trial_num orig t_max with oracle_id=$2"
        
#         # usage: EIG_Notebook.jl <total_timeout for each oracle int> <t_max_offset int> <use_orig_t_max bool> <grid_size> <trial_num> <A_oracle_num> <D_oracle_num>
#         julia "EIG_Notebook.jl" $1 "0" "true" $grid_size $trial_num $2 $2
#     done
# done

# # run the t_max_offset experiments
# for t_max_offset in $(seq 0 10 30);  # [0 10 20 30]
# do
#     echo "running t_max_offset=$t_max_offset runs"

#     for grid_size in $(seq 3 10);
#     do
#         for trial_num in $(seq 10);
#         do
#             echo "running grid_size=$grid_size trial_num=$trial_num t_max_offset=$t_max_offset with oracle_id=$2"
            
#             # usage: EIG_Notebook.jl <total_timeout for each oracle int> <t_max_offset int> <use_orig_t_max bool> <grid_size> <trial_num> <A_oracle_num> <D_oracle_num>
#             julia "EIG_Notebook.jl" $1 $t_max_offset "false" $grid_size $trial_num $2 $2
#         done
#     done
# done
