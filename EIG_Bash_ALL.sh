#!/bin/bash

# usage: bash EIG_Notebook.jl <total_timeout for each oracle>
# runs `EIG_Notebook.jl` with different parameters

# run the t_max_offset experiments
for t_max_offset in $(seq 0 10 30);  # [0 10 20 30]
do
    echo "running t_max_offset=$t_max_offset runs"

    for grid_size in $(seq 3 10);
    do
        for trial_num in $(seq 10);
        do
            echo "running grid_size=$grid_size trial_num=$trial_num t_max_offset=$t_max_offset"
            
            # usage: EIG_Notebook.jl <total_timeout for each oracle int> <t_max_offset int> <use_orig_t_max bool> <grid_size> <trial_num>
            julia "EIG_Notebook.jl" $1 $t_max_offset "false" $grid_size $trial_num
        done
    done
done

# run the original t_max experiments
echo "running original t_max runs"

for grid_size in $(seq 3 10);
do
    for trial_num in $(seq 10);
    do
        echo "running grid_size=$grid_size trial_num=$trial_num orig t_max"
        
        # usage: EIG_Notebook.jl <total_timeout for each oracle int> <t_max_offset int> <use_orig_t_max bool> <grid_size> <trial_num>
        julia "EIG_Notebook.jl" $1 "0" "true" $grid_size $trial_num
    done
done