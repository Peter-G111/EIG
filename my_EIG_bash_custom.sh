#!/bin/bash

# runs `EIG_Notebook.jl` with different parameters (customised in code in this file)
# usage: EIG_Notebook.jl


# grid 7 trial 6 with Network oracles
ult_timeout=86400  # 1 day
grid_size=7
trial_num=6
echo "running grid_size=$grid_size trial_num=$trial_num orig t_max with total_timeout=$ult_timeout"
julia "EIG_Notebook.jl" $ult_timeout "0" "true" $grid_size $trial_num "2" "2"

# grid 7 trial 7 with Network oracles
ult_timeout=86400
grid_size=7
trial_num=7
echo "running grid_size=$grid_size trial_num=$trial_num orig t_max with total_timeout=$ult_timeout"
julia "EIG_Notebook.jl" $ult_timeout "0" "true" $grid_size $trial_num "2" "2"