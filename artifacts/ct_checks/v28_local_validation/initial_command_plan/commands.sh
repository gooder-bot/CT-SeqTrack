#!/usr/bin/env bash
set -euo pipefail
# Training only; review manifest.json before execution.
cd 'D:\desktop\research\CT-SeqTrack'
python main.py --cfg 'D:\desktop\research\CT-SeqTrack\artifacts\ct_checks\v28_local_validation\initial_command_plan\configs\ct28_initial_car_b0_seed42_60ep_bs16.yaml' --path /home/lishengjie/data/nuscenes-mini --category_name Car --seed 42 --epoch 60 --batch_size 16 --tag ct28_initial_car_b0_seed42_60ep_bs16 --log_dir 'D:\desktop\research\CT-SeqTrack\artifacts\ct_checks\v28_local_validation\initial_command_plan\runs\ct28_initial_car_b0_seed42_60ep_bs16'
