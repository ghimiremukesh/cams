#!/bin/bash

# Function to handle errors
handle_error() {
  echo "Error encountered in Python script. Exiting."
  exit 1
}

# Set up the trap to call the handle_error function on any ERR signal
trap 'handle_error' ERR

start=0.1
end=1
increment=0.1
vals=($(seq $start $increment $end))
for ((t=0; t<10; t++));
do
    echo "Collecting data for time-step ${vals[$t]}"
    python collect_data_flax_latest.py --time ${vals[t]}
    echo "Now Training for time-step ${vals[$t]}"
    python train_value_jax.py --start $((t+1))
done

