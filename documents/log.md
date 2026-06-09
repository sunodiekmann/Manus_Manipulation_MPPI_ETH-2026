mkdir -p logs
(
# N sweep (hold H=16, iters=12)
for N in 2048 512 256 128; do
  for S in 0 1 2 3 4; do
    .venv/bin/python -u -m scripts.retargeting.optimize_sequence_dynamic \
      recordings/cube_roll2 --backend mjwp \
      --num-samples $N --horizon 16 --iters 12 --substeps 10 \
      --seed $S \
      --tag $(printf "sweepN_N%04d_s%d" $N $S) --force
  done
done

# H sweep (hold N=1024, iters=12)
for H in 24 20 12 8; do
  for S in 0 1 2 3 4; do
    .venv/bin/python -u -m scripts.retargeting.optimize_sequence_dynamic \
      recordings/cube_roll2 --backend mjwp \
      --num-samples 1024 --horizon $H --iters 12 --substeps 10 \
      --seed $S \
      --tag $(printf "sweepH_H%02d_s%d" $H $S) --force
  done
done

# iter sweep (hold N=1024, H=16)
for I in 16 8 4 2; do
  for S in 0 1 2 3 4; do
    .venv/bin/python -u -m scripts.retargeting.optimize_sequence_dynamic \
      recordings/cube_roll2 --backend mjwp \
      --num-samples 1024 --horizon 16 --iters $I --substeps 10 \
      --seed $S \
      --tag $(printf "sweepI_I%02d_s%d" $I $S) --force
  done
done
) 2>&1 | tee logs/pitch_sweep_$(date +%Y%m%d_%H%M).log

 
 
 
  t=   0/240  realized: pos=  9.3mm  rot=  5.3°  frac_term=0.00  rate=0.38 t/s  ETA=633s
           plan: rew=  -1.88  best_sample=  -1.90  (MPPI improv +0.75)
           per-iter rew_max:  -2.65  -2.46  -2.29  -2.15  -2.03  -2.00  -1.91  -1.90
           per-iter plan_rew: -2.64  -2.45  -2.22  -2.03  -2.01  -1.99  -1.90  -1.88
           plan terms (forecast): cube_pos= -0.12  cube_quat= -0.34  finger_qpos= -0.11  contact= -0.92  wrist_delta= -0.19  qvel= -0.13  cube_angvel= -0.07
  t=   1/240  realized: pos= 11.0mm  rot=  2.3°  frac_term=0.00  rate=0.38 t/s  ETA=627s
           plan: rew=  -1.86  best_sample=  -1.88  (MPPI improv +0.09)
           per-iter rew_max:  -1.97  -1.97  -1.97  -1.97  -1.97  -1.93  -1.90  -1.88
           per-iter plan_rew: -1.97  -1.97  -1.97  -1.97  -1.97  -1.94  -1.89  -1.86
           plan terms (forecast): cube_pos= -0.12  cube_quat= -0.27  finger_qpos= -0.11  contact= -0.94  wrist_delta= -0.19  qvel= -0.14  cube_angvel= -0.10
  t=   2/240  realized: pos= 12.4mm  rot=  1.0°  frac_term=0.00  rate=0.38 t/s  ETA=626s
           plan: rew=  -1.67  best_sample=  -1.68  (MPPI improv +0.16)
           per-iter rew_max:  -1.84  -1.85  -1.84  -1.85  -1.83  -1.76  -1.75  -1.68
           per-iter plan_rew: -1.85  -1.85  -1.85  -1.84  -1.80  -1.76  -1.71  -1.67
           plan terms (forecast): cube_pos= -0.12  cube_quat= -0.18  finger_qpos= -0.10  contact= -0.90  wrist_delta= -0.19  qvel= -0.12  cube_angvel= -0.06
  t=   3/240  realized: pos= 13.1mm  rot=  0.8°  frac_term=0.00  rate=0.38 t/s  ETA=622s
           plan: rew=  -1.61  best_sample=  -1.63  (MPPI improv +0.10)
           per-iter rew_max:  -1.73  -1.73  -1.73  -1.72  -1.72  -1.71  -1.63  -1.63
           per-iter plan_rew: -1.73  -1.73  -1.72  -1.72  -1.71  -1.69  -1.63  -1.61
           plan terms (forecast): cube_pos= -0.12  cube_quat= -0.18  finger_qpos= -0.10  contact= -0.86  wrist_delta= -0.17  qvel= -0.12  cube_angvel= -0.06
  t=   4/240  realized: pos= 13.3mm  rot=  1.2°  frac_term=0.00  rate=0.38 t/s  ETA=619s
           plan: rew=  -1.57  best_sample=  -1.58  (MPPI improv +0.08)
           per-iter rew_max:  -1.65  -1.65  -1.65  -1.65  -1.64  -1.61  -1.59  -1.58
           per-iter plan_rew: -1.65  -1.65  -1.65  -1.64  -1.63  -1.61  -1.59  -1.57
           plan terms (forecast): cube_pos= -0.12  cube_quat= -0.19  finger_qpos= -0.10  contact= -0.81  wrist_delta= -0.16  qvel= -0.12  cube_angvel= -0.08
  t=   5/240  realized: pos= 13.0mm  rot=  1.4°  frac_term=0.00  rate=0.38 t/s  ETA=617s
           plan: rew=  -1.53  best_sample=  -1.54  (MPPI improv +0.09)
           per-iter rew_max:  -1.63  -1.63  -1.63  -1.63  -1.63  -1.58  -1.56  -1.54
           per-iter plan_rew: -1.63  -1.63  -1.63  -1.63  -1.59  -1.56  -1.55  -1.53
           plan terms (forecast): cube_pos= -0.12  cube_quat= -0.18  finger_qpos= -0.10  contact= -0.78  wrist_delta= -0.16  qvel= -0.11  cube_angvel= -0.07
  t=   6/240  realized: pos= 11.5mm  rot=  0.9°  frac_term=0.00  rate=0.38 t/s  ETA=614s
           plan: rew=  -1.52  best_sample=  -1.54  (MPPI improv +0.09)
           per-iter rew_max:  -1.63  -1.63  -1.63  -1.63  -1.63  -1.62  -1.57  -1.54
           per-iter plan_rew: -1.63  -1.63  -1.63  -1.63  -1.63  -1.58  -1.55  -1.52
           plan terms (forecast): cube_pos= -0.13  cube_quat= -0.19  finger_qpos= -0.09  contact= -0.76  wrist_delta= -0.16  qvel= -0.11  cube_angvel= -0.08
  t=   7/240  realized: pos= 10.8mm  rot=  0.4°  frac_term=0.00  rate=0.38 t/s  ETA=611s
           plan: rew=  -1.51  best_sample=  -1.52  (MPPI improv +0.16)
           per-iter rew_max:  -1.68  -1.68  -1.68  -1.67  -1.67  -1.62  -1.55  -1.52
           per-iter plan_rew: -1.68  -1.68  -1.67  -1.67  -1.63  -1.57  -1.54  -1.51
           plan terms (forecast): cube_pos= -0.13  cube_quat= -0.20  finger_qpos= -0.09  contact= -0.75  wrist_delta= -0.15  qvel= -0.11  cube_angvel= -0.08
  t=   8/240  realized: pos= 11.0mm  rot=  0.4°  frac_term=0.00  rate=0.38 t/s  ETA=610s
           plan: rew=  -1.54  best_sample=  -1.56  (MPPI improv +0.18)
           per-iter rew_max:  -1.75  -1.75  -1.73  -1.71  -1.62  -1.62  -1.59  -1.56
           per-iter plan_rew: -1.76  -1.73  -1.71  -1.62  -1.62  -1.62  -1.58  -1.54
           plan terms (forecast): cube_pos= -0.14  cube_quat= -0.24  finger_qpos= -0.09  contact= -0.73  wrist_delta= -0.15  qvel= -0.11  cube_angvel= -0.09
  t=   9/240  realized: pos= 11.1mm  rot=  0.2°  frac_term=0.00  rate=0.38 t/s  ETA=611s
           plan: rew=  -1.50  best_sample=  -1.52  (MPPI improv +0.14)
           per-iter rew_max:  -1.66  -1.65  -1.65  -1.63  -1.61  -1.58  -1.55  -1.52
           per-iter plan_rew: -1.65  -1.66  -1.63  -1.61  -1.58  -1.56  -1.52  -1.50
           plan terms (forecast): cube_pos= -0.14  cube_quat= -0.25  finger_qpos= -0.09  contact= -0.68  wrist_delta= -0.14  qvel= -0.11  cube_angvel= -0.09
  t=  10/240  realized: pos= 11.4mm  rot=  0.7°  frac_term=0.00  rate=0.38 t/s  ETA=610s
           plan: rew=  -1.42  best_sample=  -1.42  (MPPI improv +0.32)
           per-iter rew_max:  -1.75  -1.73  -1.73  -1.64  -1.58  -1.54  -1.45  -1.42
           per-iter plan_rew: -1.73  -1.73  -1.71  -1.59  -1.58  -1.51  -1.45  -1.42
           plan terms (forecast): cube_pos= -0.15  cube_quat= -0.20  finger_qpos= -0.09  contact= -0.64  wrist_delta= -0.16  qvel= -0.12  cube_angvel= -0.07
  t=  11/240  realized: pos= 12.4mm  rot=  0.3°  frac_term=0.00  rate=0.37 t/s  ETA=610s
           plan: rew=  -1.34  best_sample=  -1.35  (MPPI improv +0.32)
           per-iter rew_max:  -1.66  -1.66  -1.65  -1.59  -1.48  -1.45  -1.40  -1.35
           per-iter plan_rew: -1.66  -1.66  -1.65  -1.50  -1.47  -1.42  -1.37  -1.34
           plan terms (forecast): cube_pos= -0.15  cube_quat= -0.17  finger_qpos= -0.09  contact= -0.60  wrist_delta= -0.16  qvel= -0.11  cube_angvel= -0.06
  t=  12/240  realized: pos= 12.4mm  rot=  0.2°  frac_term=0.00  rate=0.37 t/s  ETA=611s
           plan: rew=  -1.36  best_sample=  -1.37  (MPPI improv +0.38)
           per-iter rew_max:  -1.75  -1.75  -1.65  -1.57  -1.42  -1.41  -1.40  -1.37
           per-iter plan_rew: -1.75  -1.75  -1.64  -1.50  -1.42  -1.41  -1.40  -1.36
           plan terms (forecast): cube_pos= -0.17  cube_quat= -0.16  finger_qpos= -0.09  contact= -0.56  wrist_delta= -0.16  qvel= -0.14  cube_angvel= -0.08
  t=  13/240  realized: pos= 13.2mm  rot=  0.5°  frac_term=0.00  rate=0.37 t/s  ETA=611s
           plan: rew=  -1.53  best_sample=  -1.54  (MPPI improv +0.40)
           per-iter rew_max:  -1.94  -1.74  -1.75  -1.73  -1.68  -1.63  -1.57  -1.54
           per-iter plan_rew: -1.94  -1.75  -1.73  -1.73  -1.64  -1.60  -1.55  -1.53
           plan terms (forecast): cube_pos= -0.17  cube_quat= -0.24  finger_qpos= -0.10  contact= -0.54  wrist_delta= -0.20  qvel= -0.18  cube_angvel= -0.10
  t=  14/240  realized: pos= 13.3mm  rot=  0.8°  frac_term=0.00  rate=0.37 t/s  ETA=613s
           plan: rew=  -1.49  best_sample=  -1.49  (MPPI improv +0.40)
           per-iter rew_max:  -1.88  -1.88  -1.78  -1.71  -1.60  -1.56  -1.52  -1.49
           per-iter plan_rew: -1.88  -1.88  -1.79  -1.65  -1.58  -1.52  -1.49  -1.49
           plan terms (forecast): cube_pos= -0.17  cube_quat= -0.23  finger_qpos= -0.11  contact= -0.51  wrist_delta= -0.16  qvel= -0.21  cube_angvel= -0.09
  t=  15/240  realized: pos= 12.4mm  rot=  0.2°  frac_term=0.00  rate=0.36 t/s  ETA=615s
           plan: rew=  -1.67  best_sample=  -1.67  (MPPI improv +0.19)
           per-iter rew_max:  -1.86  -1.86  -1.86  -1.83  -1.78  -1.75  -1.71  -1.67
           per-iter plan_rew: -1.86  -1.86  -1.83  -1.82  -1.77  -1.76  -1.70  -1.67
           plan terms (forecast): cube_pos= -0.17  cube_quat= -0.34  finger_qpos= -0.12  contact= -0.56  wrist_delta= -0.14  qvel= -0.20  cube_angvel= -0.15
  t=  16/240  realized: pos= 13.5mm  rot=  1.0°  frac_term=0.00  rate=0.36 t/s  ETA=615s
           plan: rew=  -1.68  best_sample=  -1.69  (MPPI improv +0.31)
           per-iter rew_max:  -2.00  -1.97  -1.96  -1.96  -1.83  -1.75  -1.74  -1.69
           per-iter plan_rew: -1.97  -1.96  -2.02  -1.95  -1.82  -1.74  -1.71  -1.68
           plan terms (forecast): cube_pos= -0.18  cube_quat= -0.36  finger_qpos= -0.12  contact= -0.57  wrist_delta= -0.11  qvel= -0.20  cube_angvel= -0.14
  t=  17/240  realized: pos= 13.6mm  rot=  0.7°  frac_term=0.00  rate=0.36 t/s  ETA=615s
           plan: rew=  -1.68  best_sample=  -1.69  (MPPI improv +0.31)
           per-iter rew_max:  -2.00  -2.00  -2.00  -1.98  -1.92  -1.83  -1.76  -1.69
           per-iter plan_rew: -2.00  -2.00  -2.00  -2.01  -1.91  -1.86  -1.72  -1.68
           plan terms (forecast): cube_pos= -0.18  cube_quat= -0.38  finger_qpos= -0.12  contact= -0.55  wrist_delta= -0.10  qvel= -0.19  cube_angvel= -0.16
  t=  18/240  realized: pos= 13.3mm  rot=  0.9°  frac_term=0.00  rate=0.36 t/s  ETA=615s
           plan: rew=  -1.97  best_sample=  -1.96  (MPPI improv +0.24)
           per-iter rew_max:  -2.20  -2.20  -2.19  -2.18  -2.16  -2.03  -1.99  -1.96
           per-iter plan_rew: -2.20  -2.19  -2.18  -2.30  -2.17  -2.03  -1.99  -1.97
           plan terms (forecast): cube_pos= -0.20  cube_quat= -0.44  finger_qpos= -0.13  contact= -0.74  wrist_delta= -0.11  qvel= -0.19  cube_angvel= -0.17
  t=  19/240  realized: pos= 13.9mm  rot=  1.4°  frac_term=0.00  rate=0.36 t/s  ETA=615s
           plan: rew=  -1.98  best_sample=  -1.97  (MPPI improv +0.40)
           per-iter rew_max:  -2.37  -2.40  -2.36  -2.25  -2.16  -2.05  -2.02  -1.97
           per-iter plan_rew: -2.40  -2.38  -2.27  -2.19  -2.12  -2.05  -2.01  -1.98
           plan terms (forecast): cube_pos= -0.21  cube_quat= -0.39  finger_qpos= -0.13  contact= -0.81  wrist_delta= -0.09  qvel= -0.19  cube_angvel= -0.17
  t=  20/240  realized: pos= 16.3mm  rot=  1.4°  frac_term=0.00  rate=0.36 t/s  ETA=615s
           plan: rew=  -2.10  best_sample=  -2.11  (MPPI improv +0.35)
           per-iter rew_max:  -2.46  -2.41  -2.36  -2.26  -2.21  -2.17  -2.14  -2.11
           per-iter plan_rew: -2.41  -2.37  -2.33  -2.22  -2.18  -2.14  -2.13  -2.10
           plan terms (forecast): cube_pos= -0.22  cube_quat= -0.39  finger_qpos= -0.14  contact= -0.89  wrist_delta= -0.10  qvel= -0.20  cube_angvel= -0.17
  t=  21/240  realized: pos= 19.6mm  rot=  1.0°  frac_term=0.00  rate=0.35 t/s  ETA=615s
           plan: rew=  -2.08  best_sample=  -2.10  (MPPI improv +0.39)
           per-iter rew_max:  -2.50  -2.44  -2.32  -2.30  -2.23  -2.20  -2.14  -2.10
           per-iter plan_rew: -2.50  -2.32  -2.32  -2.29  -2.23  -2.21  -2.13  -2.08
           plan terms (forecast): cube_pos= -0.22  cube_quat= -0.32  finger_qpos= -0.15  contact= -0.95  wrist_delta= -0.11  qvel= -0.20  cube_angvel= -0.13
  t=  22/240  realized: pos= 22.8mm  rot=  1.8°  frac_term=0.00  rate=0.35 t/s  ETA=613s
           plan: rew=  -2.35  best_sample=  -2.35  (MPPI improv +0.29)
           per-iter rew_max:  -2.64  -2.55  -2.47  -2.47  -2.45  -2.42  -2.39  -2.35
           per-iter plan_rew: -2.57  -2.54  -2.47  -2.47  -2.44  -2.42  -2.38  -2.35
           plan terms (forecast): cube_pos= -0.23  cube_quat= -0.41  finger_qpos= -0.16  contact= -1.02  wrist_delta= -0.13  qvel= -0.21  cube_angvel= -0.20
  t=  23/240  realized: pos= 25.8mm  rot=  0.8°  frac_term=0.00  rate=0.35 t/s  ETA=613s
           plan: rew=  -2.42  best_sample=  -2.44  (MPPI improv +0.36)
           per-iter rew_max:  -2.79  -2.78  -2.74  -2.67  -2.59  -2.48  -2.47  -2.44
           per-iter plan_rew: -2.80  -2.81  -2.72  -2.65  -2.53  -2.49  -2.48  -2.42
           plan terms (forecast): cube_pos= -0.24  cube_quat= -0.42  finger_qpos= -0.17  contact= -1.08  wrist_delta= -0.13  qvel= -0.22  cube_angvel= -0.17
  t=  24/240  realized: pos= 24.5mm  rot=  0.9°  frac_term=0.00  rate=0.35 t/s  ETA=614s
           plan: rew=  -2.39  best_sample=  -2.40  (MPPI improv +0.47)
           per-iter rew_max:  -2.87  -2.79  -2.60  -2.54  -2.53  -2.47  -2.42  -2.40
           per-iter plan_rew: -2.79  -2.73  -2.84  -2.53  -2.53  -2.48  -2.44  -2.39
           plan terms (forecast): cube_pos= -0.22  cube_quat= -0.37  finger_qpos= -0.17  contact= -1.10  wrist_delta= -0.14  qvel= -0.22  cube_angvel= -0.16
  t=  25/240  realized: pos= 23.5mm  rot=  1.4°  frac_term=0.00  rate=0.35 t/s  ETA=616s
           plan: rew=  -2.59  best_sample=  -2.59  (MPPI improv +0.22)
           per-iter rew_max:  -2.82  -2.80  -2.81  -2.76  -2.76  -2.72  -2.65  -2.59
           per-iter plan_rew: -2.84  -2.81  -2.81  -2.76  -2.76  -2.71  -2.67  -2.59
           plan terms (forecast): cube_pos= -0.23  cube_quat= -0.48  finger_qpos= -0.18  contact= -1.15  wrist_delta= -0.13  qvel= -0.23  cube_angvel= -0.20
  t=  26/240  realized: pos= 23.6mm  rot=  1.4°  frac_term=0.00  rate=0.34 t/s  ETA=618s
           plan: rew=  -2.63  best_sample=  -2.62  (MPPI improv +0.32)
           per-iter rew_max:  -2.94  -2.89  -2.89  -2.83  -2.73  -2.69  -2.68  -2.62
           per-iter plan_rew: -2.89  -2.89  -2.86  -2.77  -2.75  -2.70  -2.67  -2.63
           plan terms (forecast): cube_pos= -0.23  cube_quat= -0.50  finger_qpos= -0.17  contact= -1.18  wrist_delta= -0.14  qvel= -0.23  cube_angvel= -0.18
  t=  27/240  realized: pos= 22.4mm  rot=  1.8°  frac_term=0.00  rate=0.34 t/s  ETA=621s
           plan: rew=  -2.75  best_sample=  -2.71  (MPPI improv +0.27)
           per-iter rew_max:  -2.98  -2.99  -2.96  -2.93  -2.82  -2.78  -2.75  -2.71
           per-iter plan_rew: -3.01  -2.97  -2.96  -2.96  -2.83  -2.80  -2.74  -2.75
           plan terms (forecast): cube_pos= -0.22  cube_quat= -0.54  finger_qpos= -0.18  contact= -1.26  wrist_delta= -0.13  qvel= -0.23  cube_angvel= -0.18
  t=  28/240  realized: pos= 18.5mm  rot=  1.7°  frac_term=0.00  rate=0.34 t/s  ETA=623s
           plan: rew=  -2.85  best_sample=  -2.79  (MPPI improv +0.23)
           per-iter rew_max:  -3.01  -3.01  -3.00  -3.00  -2.89  -2.87  -2.83  -2.79
           per-iter plan_rew: -3.02  -3.07  -3.03  -2.90  -2.87  -2.86  -2.82  -2.85
           plan terms (forecast): cube_pos= -0.23  cube_quat= -0.61  finger_qpos= -0.18  contact= -1.32  wrist_delta= -0.13  qvel= -0.21  cube_angvel= -0.18
  t=  29/240  realized: pos= 15.8mm  rot=  3.4°  frac_term=0.00  rate=0.34 t/s  ETA=625s
           plan: rew=  -2.86  best_sample=  -2.85  (MPPI improv +0.30)
           per-iter rew_max:  -3.16  -3.14  -3.16  -3.04  -2.97  -2.90  -2.87  -2.85
           per-iter plan_rew: -3.16  -3.17  -3.11  -3.03  -2.90  -2.90  -2.88  -2.86
           plan terms (forecast): cube_pos= -0.23  cube_quat= -0.63  finger_qpos= -0.18  contact= -1.36  wrist_delta= -0.11  qvel= -0.17  cube_angvel= -0.19
  t=  30/240  realized: pos= 15.4mm  rot=  5.5°  frac_term=0.00  rate=0.33 t/s  ETA=627s
           plan: rew=  -2.90  best_sample=  -2.89  (MPPI improv +0.45)
           per-iter rew_max:  -3.34  -3.31  -3.19  -3.16  -3.07  -2.98  -2.95  -2.89
           per-iter plan_rew: -3.35  -3.24  -3.21  -3.11  -3.08  -3.00  -2.95  -2.90
           plan terms (forecast): cube_pos= -0.22  cube_quat= -0.56  finger_qpos= -0.18  contact= -1.43  wrist_delta= -0.16  qvel= -0.18  cube_angvel= -0.17
  t=  31/240  realized: pos= 20.8mm  rot=  0.9°  frac_term=0.00  rate=0.33 t/s  ETA=629s
           plan: rew=  -3.24  best_sample=  -3.25  (MPPI improv +0.99)
           per-iter rew_max:  -4.24  -3.57  -3.52  -3.49  -3.41  -3.33  -3.27  -3.25
           per-iter plan_rew: -4.23  -3.55  -3.50  -3.52  -3.45  -3.31  -3.27  -3.24
           plan terms (forecast): cube_pos= -0.28  cube_quat= -0.44  finger_qpos= -0.23  contact= -1.65  wrist_delta= -0.24  qvel= -0.25  cube_angvel= -0.14
  t=  32/240  realized: pos= 23.0mm  rot=  0.7°  frac_term=0.00  rate=0.33 t/s  ETA=630s
           plan: rew=  -3.47  best_sample=  -3.45  (MPPI improv +0.35)
           per-iter rew_max:  -3.80  -3.69  -3.67  -3.67  -3.65  -3.55  -3.52  -3.45
           per-iter plan_rew: -3.69  -3.71  -3.67  -3.67  -3.63  -3.56  -3.51  -3.47
           plan terms (forecast): cube_pos= -0.28  cube_quat= -0.56  finger_qpos= -0.22  contact= -1.70  wrist_delta= -0.24  qvel= -0.28  cube_angvel= -0.19
  t=  33/240  realized: pos= 24.0mm  rot=  1.3°  frac_term=0.00  rate=0.33 t/s  ETA=631s
           plan: rew=  -3.35  best_sample=  -3.32  (MPPI improv +0.79)
           per-iter rew_max:  -4.11  -4.03  -3.83  -3.56  -3.52  -3.40  -3.36  -3.32
           per-iter plan_rew: -4.08  -4.10  -3.81  -3.55  -3.55  -3.37  -3.42  -3.35
           plan terms (forecast): cube_pos= -0.31  cube_quat= -0.46  finger_qpos= -0.24  contact= -1.74  wrist_delta= -0.22  qvel= -0.26  cube_angvel= -0.12
  t=  34/240  realized: pos= 25.7mm  rot=  0.2°  frac_term=0.00  rate=0.32 t/s  ETA=633s
           plan: rew=  -3.41  best_sample=  -3.40  (MPPI improv +0.77)
           per-iter rew_max:  -4.17  -3.90  -3.81  -3.57  -3.54  -3.40  -3.45  -3.40
           per-iter plan_rew: -6.42  -3.84  -3.89  -3.54  -3.52  -3.49  -3.45  -3.41
           plan terms (forecast): cube_pos= -0.31  cube_quat= -0.50  finger_qpos= -0.24  contact= -1.68  wrist_delta= -0.22  qvel= -0.30  cube_angvel= -0.15
  t=  35/240  realized: pos= 27.8mm  rot=  0.4°  frac_term=0.00  rate=0.32 t/s  ETA=634s
           plan: rew=  -4.26  best_sample=  -3.41  (MPPI improv +0.08)
           per-iter rew_max:  -3.49  -3.50  -3.50  -3.50  -3.53  -3.49  -3.42  -3.41
           per-iter plan_rew: -3.58  -3.50  -3.53  -3.99  -4.11  -3.61  -3.81  -4.26
           plan terms (forecast): cube_pos= -0.42  cube_quat= -0.80  finger_qpos= -0.23  contact= -2.08  wrist_delta= -0.20  qvel= -0.29  cube_angvel= -0.25
  t=  36/240  realized: pos= 28.3mm  rot=  0.8°  frac_term=0.00  rate=0.32 t/s  ETA=634s
           plan: rew=  -3.85  best_sample=  -3.83  (MPPI improv +0.01)
           per-iter rew_max:  -3.84  -3.85  -3.85  -3.89  -3.85  -3.83  -3.83  -3.83
           per-iter plan_rew: -3.86  -3.88  -3.88  -3.85  -3.88  -3.84  -4.68  -3.85
           plan terms (forecast): cube_pos= -0.45  cube_quat= -0.49  finger_qpos= -0.22  contact= -2.11  wrist_delta= -0.19  qvel= -0.27  cube_angvel= -0.13
  t=  37/240  realized: pos= 31.1mm  rot=  2.0°  frac_term=0.00  rate=0.32 t/s  ETA=634s
           plan: rew=  -4.76  best_sample=  -4.65  (MPPI improv +1.24)
           per-iter rew_max:  -5.89  -5.07  -4.96  -4.86  -4.83  -4.72  -4.67  -4.65
           per-iter plan_rew: -5.66  -5.07  -4.95  -5.41  -4.81  -4.70  -10.73  -4.76
           plan terms (forecast): cube_pos= -0.29  cube_quat= -1.89  finger_qpos= -0.22  contact= -1.57  wrist_delta= -0.22  qvel= -0.28  cube_angvel= -0.30
  t=  38/240  realized: pos= 33.3mm  rot=  3.6°  frac_term=0.00  rate=0.32 t/s  ETA=632s
           plan: rew= -14.07  best_sample=  -4.44  (MPPI improv +0.56)
           per-iter rew_max:  -5.00  -4.86  -4.85  -4.77  -4.74  -4.56  -4.47  -4.44
           per-iter plan_rew: -12.72  -4.99  -4.85  -4.90  -4.81  -11.23  -4.60  -14.07
           plan terms (forecast): cube_pos= -0.66  cube_quat= -8.59  finger_qpos= -0.22  contact= -3.17  wrist_delta= -0.21  qvel= -0.29  cube_angvel= -0.93
  t=  39/240  realized: pos= 31.6mm  rot=  1.5°  frac_term=0.00  rate=0.32 t/s  ETA=630s
           plan: rew=  -4.40  best_sample=  -4.40  (MPPI improv +0.58)
           per-iter rew_max:  -4.97  -4.97  -5.14  -5.02  -4.85  -4.63  -4.50  -4.40
           per-iter plan_rew: -4.97  -5.82  -5.83  -5.38  -4.91  -4.70  -4.57  -4.40
           plan terms (forecast): cube_pos= -0.32  cube_quat= -1.50  finger_qpos= -0.22  contact= -1.50  wrist_delta= -0.24  qvel= -0.31  cube_angvel= -0.31
  t=  40/240  realized: pos= 27.4mm  rot=  2.1°  frac_term=0.00  rate=0.32 t/s  ETA=628s
           plan: rew=  -4.56  best_sample=  -4.54  (MPPI improv +0.20)
           per-iter rew_max:  -4.75  -4.73  -4.86  -4.72  -4.72  -4.66  -4.59  -4.54
           per-iter plan_rew: -4.74  -4.91  -4.72  -4.89  -4.75  -4.60  -4.59  -4.56
           plan terms (forecast): cube_pos= -0.32  cube_quat= -1.70  finger_qpos= -0.21  contact= -1.46  wrist_delta= -0.23  qvel= -0.30  cube_angvel= -0.34
  t=  41/240  realized: pos= 26.6mm  rot=  2.3°  frac_term=0.00  rate=0.32 t/s  ETA=625s
           plan: rew=  -4.50  best_sample=  -4.52  (MPPI improv +0.37)
           per-iter rew_max:  -4.89  -4.85  -4.78  -4.72  -4.63  -4.59  -4.53  -4.52
           per-iter plan_rew: -4.87  -4.87  -4.76  -4.74  -16.78  -4.59  -4.53  -4.50
           plan terms (forecast): cube_pos= -0.32  cube_quat= -1.73  finger_qpos= -0.20  contact= -1.42  wrist_delta= -0.22  qvel= -0.29  cube_angvel= -0.32
  t=  42/240  realized: pos= 30.1mm  rot=  0.4°  frac_term=0.00  rate=0.32 t/s  ETA=622s
           plan: rew=  -4.77  best_sample=  -4.66  (MPPI improv +0.28)
           per-iter rew_max:  -4.94  -4.96  -4.96  -4.86  -4.80  -4.72  -4.73  -4.66
           per-iter plan_rew: -10.08  -4.96  -4.86  -4.87  -4.87  -4.86  -16.40  -4.77
           plan terms (forecast): cube_pos= -0.30  cube_quat= -1.98  finger_qpos= -0.19  contact= -1.40  wrist_delta= -0.23  qvel= -0.32  cube_angvel= -0.35
  t=  43/240  realized: pos= 35.0mm  rot=  2.7°  frac_term=0.00  rate=0.32 t/s  ETA=619s
           plan: rew=  -4.79  best_sample=  -4.75  (MPPI improv +0.25)
           per-iter rew_max:  -5.00  -4.97  -4.99  -4.99  -4.83  -4.79  -4.77  -4.75
           per-iter plan_rew: -5.08  -4.99  -4.99  -4.94  -4.92  -4.77  -4.91  -4.79
           plan terms (forecast): cube_pos= -0.31  cube_quat= -2.09  finger_qpos= -0.18  contact= -1.36  wrist_delta= -0.22  qvel= -0.31  cube_angvel= -0.31
  t=  44/240  realized: pos= 39.5mm  rot=  3.6°  frac_term=0.00  rate=0.32 t/s  ETA=616s
           plan: rew=  -4.76  best_sample=  -4.77  (MPPI improv +0.30)
           per-iter rew_max:  -5.07  -5.09  -5.03  -4.79  -4.90  -4.87  -4.78  -4.77
           per-iter plan_rew: -5.18  -5.19  -5.09  -4.92  -4.98  -4.79  -4.83  -4.76
           plan terms (forecast): cube_pos= -0.30  cube_quat= -2.14  finger_qpos= -0.18  contact= -1.32  wrist_delta= -0.24  qvel= -0.29  cube_angvel= -0.30
  t=  45/240  realized: pos= 43.8mm  rot=  2.9°  frac_term=0.00  rate=0.32 t/s  ETA=613s
           plan: rew=  -4.61  best_sample=  -4.70  (MPPI improv +0.58)
           per-iter rew_max:  -5.29  -5.24  -5.25  -5.06  -5.01  -4.84  -4.70  -4.70
           per-iter plan_rew: -5.28  -5.25  -5.26  -5.07  -19.27  -5.16  -4.79  -4.61
           plan terms (forecast): cube_pos= -0.29  cube_quat= -2.05  finger_qpos= -0.17  contact= -1.28  wrist_delta= -0.22  qvel= -0.30  cube_angvel= -0.30
  t=  46/240  realized: pos= 45.2mm  rot=  1.2°  frac_term=0.00  rate=0.32 t/s  ETA=610s
           plan: rew=  -4.66  best_sample=  -4.62  (MPPI improv +1.86)
           per-iter rew_max:  -6.48  -5.58  -5.20  -5.11  -5.02  -4.72  -4.67  -4.62
           per-iter plan_rew: -6.47  -5.72  -5.49  -5.45  -5.30  -4.77  -4.67  -4.66
           plan terms (forecast): cube_pos= -0.29  cube_quat= -1.86  finger_qpos= -0.19  contact= -1.27  wrist_delta= -0.33  qvel= -0.31  cube_angvel= -0.40
  t=  47/240  realized: pos= 41.5mm  rot=  2.8°  frac_term=0.00  rate=0.32 t/s  ETA=607s
           plan: rew=  -4.51  best_sample=  -4.53  (MPPI improv +1.14)
           per-iter rew_max:  -5.68  -5.48  -4.79  -4.82  -4.77  -4.66  -4.67  -4.53
           per-iter plan_rew: -5.64  -8.56  -4.90  -4.79  -4.75  -4.67  -4.75  -4.51
           plan terms (forecast): cube_pos= -0.30  cube_quat= -1.71  finger_qpos= -0.19  contact= -1.28  wrist_delta= -0.33  qvel= -0.31  cube_angvel= -0.39
  t=  48/240  realized: pos= 34.7mm  rot=  8.2°  frac_term=0.00  rate=0.32 t/s  ETA=604s
           plan: rew= -20.59  best_sample=  -5.05  (MPPI improv +1.00)
           per-iter rew_max:  -6.05  -5.82  -5.36  -5.36  -5.15  -5.07  -5.01  -5.05
           per-iter plan_rew: -6.38  -6.11  -5.36  -8.96  -6.19  -5.04  -5.05  -20.59
           plan terms (forecast): cube_pos= -1.02  cube_quat=-13.14  finger_qpos= -0.18  contact= -4.61  wrist_delta= -0.30  qvel= -0.35  cube_angvel= -0.98
  t=  49/240  realized: pos= 28.5mm  rot= 13.4°  frac_term=0.00  rate=0.32 t/s  ETA=602s
           plan: rew= -28.72  best_sample=  -4.65  (MPPI improv +1.08)
           per-iter rew_max:  -5.73  -5.67  -5.34  -5.11  -4.87  -4.73  -4.64  -4.65
           per-iter plan_rew: -6.31  -5.83  -5.64  -5.80  -5.58  -4.71  -4.90  -28.72
           plan terms (forecast): cube_pos= -1.06  cube_quat=-19.80  finger_qpos= -0.20  contact= -5.98  wrist_delta= -0.26  qvel= -0.35  cube_angvel= -1.07
  t=  50/240  realized: pos= 28.1mm  rot= 17.8°  frac_term=0.00  rate=0.32 t/s  ETA=600s
           plan: rew=  -4.31  best_sample=  -4.32  (MPPI improv +0.48)
           per-iter rew_max:  -4.80  -4.69  -4.80  -4.43  -4.42  -4.41  -4.36  -4.32
           per-iter plan_rew: -4.79  -4.89  -4.93  -4.49  -4.41  -4.41  -4.32  -4.31
           plan terms (forecast): cube_pos= -0.26  cube_quat= -1.70  finger_qpos= -0.19  contact= -1.21  wrist_delta= -0.27  qvel= -0.36  cube_angvel= -0.32
  t=  51/240  realized: pos= 28.8mm  rot= 16.5°  frac_term=0.00  rate=0.32 t/s  ETA=596s
           plan: rew=  -4.05  best_sample=  -4.06  (MPPI improv +0.82)
           per-iter rew_max:  -4.88  -4.56  -4.51  -4.51  -4.41  -4.17  -4.10  -4.06
           per-iter plan_rew: -4.88  -25.13  -5.06  -4.88  -5.68  -4.20  -4.11  -4.05
           plan terms (forecast): cube_pos= -0.30  cube_quat= -1.42  finger_qpos= -0.20  contact= -1.17  wrist_delta= -0.31  qvel= -0.33  cube_angvel= -0.31
  t=  52/240  realized: pos= 27.7mm  rot= 12.0°  frac_term=0.00  rate=0.32 t/s  ETA=593s
           plan: rew=  -4.77  best_sample=  -3.76  (MPPI improv +1.20)
           per-iter rew_max:  -4.96  -4.81  -4.48  -4.35  -4.14  -3.91  -3.83  -3.76
           per-iter plan_rew: -5.89  -5.70  -4.90  -4.30  -4.11  -3.94  -4.19  -4.77
           plan terms (forecast): cube_pos= -0.30  cube_quat= -1.99  finger_qpos= -0.24  contact= -1.24  wrist_delta= -0.27  qvel= -0.36  cube_angvel= -0.37
  t=  53/240  realized: pos= 27.1mm  rot= 10.8°  frac_term=0.00  rate=0.32 t/s  ETA=590s
           plan: rew=  -4.08  best_sample=  -4.10  (MPPI improv +0.73)
           per-iter rew_max:  -4.83  -4.75  -4.47  -4.71  -4.48  -4.35  -4.20  -4.10
           per-iter plan_rew: -6.28  -4.68  -6.13  -4.61  -4.45  -4.34  -4.29  -4.08
           plan terms (forecast): cube_pos= -0.31  cube_quat= -1.31  finger_qpos= -0.21  contact= -1.29  wrist_delta= -0.26  qvel= -0.37  cube_angvel= -0.34
  t=  54/240  realized: pos= 31.4mm  rot= 10.7°  frac_term=0.00  rate=0.31 t/s  ETA=588s
           plan: rew=  -4.39  best_sample=  -3.88  (MPPI improv +1.88)
           per-iter rew_max:  -5.76  -4.96  -4.64  -4.08  -4.06  -4.01  -3.94  -3.88
           per-iter plan_rew: -5.50  -4.97  -4.64  -4.40  -6.84  -4.15  -3.91  -4.39
           plan terms (forecast): cube_pos= -0.32  cube_quat= -1.62  finger_qpos= -0.24  contact= -1.32  wrist_delta= -0.32  qvel= -0.34  cube_angvel= -0.23
  t=  55/240  realized: pos= 32.9mm  rot= 10.0°  frac_term=0.00  rate=0.31 t/s  ETA=585s
           plan: rew=  -3.36  best_sample=  -3.41  (MPPI improv +1.12)
           per-iter rew_max:  -4.53  -4.04  -4.10  -3.59  -3.48  -3.52  -3.44  -3.41
           per-iter plan_rew: -4.80  -35.57  -4.02  -3.69  -3.52  -3.51  -3.73  -3.36
           plan terms (forecast): cube_pos= -0.26  cube_quat= -0.80  finger_qpos= -0.24  contact= -1.26  wrist_delta= -0.25  qvel= -0.36  cube_angvel= -0.18
  t=  56/240  realized: pos= 34.9mm  rot=  9.2°  frac_term=0.00  rate=0.31 t/s  ETA=583s
           plan: rew=  -4.10  best_sample=  -3.27  (MPPI improv +0.50)
           per-iter rew_max:  -3.77  -4.05  -3.69  -3.46  -3.38  -3.27  -3.27  -3.27
           per-iter plan_rew: -4.18  -4.18  -4.53  -3.82  -3.93  -3.27  -3.50  -4.10
           plan terms (forecast): cube_pos= -0.26  cube_quat= -1.33  finger_qpos= -0.23  contact= -1.40  wrist_delta= -0.23  qvel= -0.34  cube_angvel= -0.31
  t=  57/240  realized: pos= 36.7mm  rot=  5.2°  frac_term=0.00  rate=0.31 t/s  ETA=580s
           plan: rew=  -3.31  best_sample=  -3.30  (MPPI improv +0.71)
           per-iter rew_max:  -4.01  -3.69  -3.48  -3.48  -3.43  -3.36  -3.33  -3.30
           per-iter plan_rew: -3.69  -3.73  -3.48  -3.71  -3.46  -3.33  -3.36  -3.31
           plan terms (forecast): cube_pos= -0.26  cube_quat= -0.72  finger_qpos= -0.23  contact= -1.23  wrist_delta= -0.29  qvel= -0.33  cube_angvel= -0.26
  t=  58/240  realized: pos= 37.5mm  rot=  6.8°  frac_term=0.00  rate=0.31 t/s  ETA=576s
           plan: rew=  -3.39  best_sample=  -3.09  (MPPI improv +0.38)
           per-iter rew_max:  -3.47  -3.47  -3.33  -3.18  -3.17  -3.17  -3.15  -3.09
           per-iter plan_rew: -3.47  -3.46  -3.45  -3.18  -3.17  -3.17  -3.25  -3.39
           plan terms (forecast): cube_pos= -0.24  cube_quat= -0.88  finger_qpos= -0.22  contact= -1.23  wrist_delta= -0.26  qvel= -0.31  cube_angvel= -0.24
  t=  59/240  realized: pos= 36.3mm  rot=  4.0°  frac_term=0.00  rate=0.31 t/s  ETA=573s
           plan: rew=  -2.88  best_sample=  -2.86  (MPPI improv +0.39)
           per-iter rew_max:  -3.25  -3.22  -3.29  -3.16  -3.06  -3.00  -2.97  -2.86
           per-iter plan_rew: -3.47  -3.29  -3.64  -3.51  -3.07  -3.07  -2.95  -2.88
           plan terms (forecast): cube_pos= -0.23  cube_quat= -0.55  finger_qpos= -0.21  contact= -1.11  wrist_delta= -0.23  qvel= -0.30  cube_angvel= -0.24
  t=  60/240  realized: pos= 35.4mm  rot=  2.3°  frac_term=0.00  rate=0.31 t/s  ETA=570s
           plan: rew=  -3.05  best_sample=  -2.85  (MPPI improv +0.68)
           per-iter rew_max:  -3.53  -3.25  -3.29  -3.12  -3.03  -2.93  -2.90  -2.85
           per-iter plan_rew: -5.09  -3.29  -3.31  -3.13  -3.04  -2.93  -2.89  -3.05
           plan terms (forecast): cube_pos= -0.20  cube_quat= -0.77  finger_qpos= -0.22  contact= -1.11  wrist_delta= -0.22  qvel= -0.34  cube_angvel= -0.19
  t=  61/240  realized: pos= 35.7mm  rot=  1.7°  frac_term=0.00  rate=0.31 t/s  ETA=567s
           plan: rew=  -2.64  best_sample=  -2.66  (MPPI improv +0.89)
           per-iter rew_max:  -3.54  -3.31  -3.15  -3.02  -2.86  -2.77  -2.70  -2.66
           per-iter plan_rew: -3.36  -3.51  -3.03  -3.13  -3.16  -2.80  -2.70  -2.64
           plan terms (forecast): cube_pos= -0.18  cube_quat= -0.46  finger_qpos= -0.23  contact= -1.05  wrist_delta= -0.22  qvel= -0.34  cube_angvel= -0.16
  t=  62/240  realized: pos= 36.5mm  rot=  1.4°  frac_term=0.00  rate=0.31 t/s  ETA=563s
           plan: rew=  -2.93  best_sample=  -2.93  (MPPI improv +0.46)
           per-iter rew_max:  -3.39  -3.39  -3.39  -3.24  -3.03  -2.98  -2.98  -2.93
           per-iter plan_rew: -3.48  -3.48  -3.36  -3.05  -3.07  -3.03  -3.00  -2.93
           plan terms (forecast): cube_pos= -0.19  cube_quat= -0.51  finger_qpos= -0.22  contact= -1.21  wrist_delta= -0.20  qvel= -0.35  cube_angvel= -0.26
  t=  63/240  realized: pos= 35.8mm  rot=  0.8°  frac_term=0.00  rate=0.31 t/s  ETA=560s
           plan: rew=  -3.01  best_sample=  -3.01  (MPPI improv +0.86)
           per-iter rew_max:  -3.86  -3.92  -3.85  -3.39  -3.21  -3.20  -3.15  -3.01
           per-iter plan_rew: -3.93  -3.94  -3.72  -3.32  -3.21  -3.23  -3.48  -3.01
           plan terms (forecast): cube_pos= -0.16  cube_quat= -0.63  finger_qpos= -0.22  contact= -1.17  wrist_delta= -0.19  qvel= -0.35  cube_angvel= -0.30
  t=  64/240  realized: pos= 31.7mm  rot=  3.0°  frac_term=0.00  rate=0.31 t/s  ETA=556s
           plan: rew=  -3.66  best_sample=  -3.24  (MPPI improv +0.70)
           per-iter rew_max:  -3.94  -3.95  -3.82  -3.65  -3.58  -3.41  -3.33  -3.24
           per-iter plan_rew: -3.95  -3.94  -4.43  -3.71  -3.51  -3.38  -3.49  -3.66
           plan terms (forecast): cube_pos= -0.17  cube_quat= -1.10  finger_qpos= -0.22  contact= -1.30  wrist_delta= -0.14  qvel= -0.33  cube_angvel= -0.40
  t=  65/240  realized: pos= 29.6mm  rot=  4.4°  frac_term=0.00  rate=0.31 t/s  ETA=553s
           plan: rew=  -3.46  best_sample=  -3.25  (MPPI improv +1.21)
           per-iter rew_max:  -4.46  -3.78  -3.70  -3.49  -3.51  -3.40  -3.29  -3.25
           per-iter plan_rew: -3.78  -3.70  -3.58  -3.57  -3.72  -3.40  -3.29  -3.46
           plan terms (forecast): cube_pos= -0.16  cube_quat= -1.08  finger_qpos= -0.22  contact= -1.18  wrist_delta= -0.14  qvel= -0.35  cube_angvel= -0.33
  t=  66/240  realized: pos= 25.7mm  rot=  3.8°  frac_term=0.00  rate=0.31 t/s  ETA=549s
           plan: rew=  -3.02  best_sample=  -3.03  (MPPI improv +0.78)
           per-iter rew_max:  -3.81  -3.45  -3.38  -3.38  -3.15  -3.14  -3.05  -3.03
           per-iter plan_rew: -3.60  -3.39  -3.53  -3.22  -3.46  -3.15  -4.52  -3.02
           plan terms (forecast): cube_pos= -0.17  cube_quat= -0.64  finger_qpos= -0.22  contact= -1.21  wrist_delta= -0.16  qvel= -0.35  cube_angvel= -0.26
  t=  67/240  realized: pos= 22.7mm  rot=  1.7°  frac_term=0.00  rate=0.31 t/s  ETA=546s
           plan: rew=  -2.92  best_sample=  -2.94  (MPPI improv +0.66)
           per-iter rew_max:  -3.60  -3.58  -3.44  -3.31  -3.14  -3.08  -2.99  -2.94
           per-iter plan_rew: -3.59  -6.65  -3.65  -3.33  -3.24  -3.08  -2.98  -2.92
           plan terms (forecast): cube_pos= -0.16  cube_quat= -0.60  finger_qpos= -0.22  contact= -1.21  wrist_delta= -0.18  qvel= -0.35  cube_angvel= -0.19
  t=  68/240  realized: pos= 18.4mm  rot=  2.9°  frac_term=0.00  rate=0.31 t/s  ETA=543s
           plan: rew=  -2.84  best_sample=  -2.83  (MPPI improv +0.60)
           per-iter rew_max:  -3.43  -3.40  -3.26  -3.09  -3.03  -2.97  -2.91  -2.83
           per-iter plan_rew: -3.40  -3.39  -3.11  -3.11  -2.98  -2.96  -2.89  -2.84
           plan terms (forecast): cube_pos= -0.15  cube_quat= -0.48  finger_qpos= -0.21  contact= -1.31  wrist_delta= -0.15  qvel= -0.34  cube_angvel= -0.21
  t=  69/240  realized: pos= 18.4mm  rot=  2.8°  frac_term=0.00  rate=0.31 t/s  ETA=540s
           plan: rew=  -2.96  best_sample=  -2.92  (MPPI improv +0.86)
           per-iter rew_max:  -3.77  -3.30  -3.34  -3.23  -3.09  -3.07  -3.00  -2.92
           per-iter plan_rew: -3.95  -7.00  -7.06  -3.26  -3.12  -3.05  -3.00  -2.96
           plan terms (forecast): cube_pos= -0.15  cube_quat= -0.49  finger_qpos= -0.21  contact= -1.39  wrist_delta= -0.20  qvel= -0.32  cube_angvel= -0.20
  t=  70/240  realized: pos= 17.6mm  rot=  1.5°  frac_term=0.00  rate=0.31 t/s  ETA=537s
           plan: rew=  -3.14  best_sample=  -3.15  (MPPI improv +0.30)
           per-iter rew_max:  -3.45  -3.45  -3.37  -3.28  -3.26  -3.28  -3.19  -3.15
           per-iter plan_rew: -3.47  -3.45  -3.28  -3.26  -3.28  -3.26  -3.19  -3.14
           plan terms (forecast): cube_pos= -0.15  cube_quat= -0.58  finger_qpos= -0.20  contact= -1.48  wrist_delta= -0.16  qvel= -0.33  cube_angvel= -0.24
  t=  71/240  realized: pos= 19.2mm  rot=  1.4°  frac_term=0.00  rate=0.31 t/s  ETA=535s
           plan: rew=  -3.10  best_sample=  -3.09  (MPPI improv +0.48)
           per-iter rew_max:  -3.56  -3.58  -3.33  -3.26  -3.22  -3.21  -3.12  -3.09
           per-iter plan_rew: -3.63  -3.54  -3.25  -3.24  -3.24  -3.18  -3.09  -3.10
           plan terms (forecast): cube_pos= -0.16  cube_quat= -0.52  finger_qpos= -0.20  contact= -1.54  wrist_delta= -0.17  qvel= -0.32  cube_angvel= -0.19
  t=  72/240  realized: pos= 20.6mm  rot=  1.4°  frac_term=0.00  rate=0.31 t/s  ETA=532s
           plan: rew=  -3.61  best_sample=  -3.28  (MPPI improv +0.41)
           per-iter rew_max:  -3.70  -3.73  -3.56  -3.50  -3.45  -3.37  -3.36  -3.28
           per-iter plan_rew: -3.74  -3.73  -3.58  -3.48  -3.37  -3.37  -3.31  -3.61
           plan terms (forecast): cube_pos= -0.17  cube_quat= -0.84  finger_qpos= -0.19  contact= -1.61  wrist_delta= -0.19  qvel= -0.32  cube_angvel= -0.30
  t=  73/240  realized: pos= 22.4mm  rot=  2.0°  frac_term=0.00  rate=0.31 t/s  ETA=530s
           plan: rew=  -3.72  best_sample=  -3.50  (MPPI improv +1.09)
           per-iter rew_max:  -4.59  -4.08  -3.90  -3.69  -3.66  -3.59  -3.53  -3.50
           per-iter plan_rew: -4.62  -4.07  -4.08  -3.74  -3.65  -4.17  -4.48  -3.72
           plan terms (forecast): cube_pos= -0.18  cube_quat= -0.82  finger_qpos= -0.20  contact= -1.71  wrist_delta= -0.22  qvel= -0.31  cube_angvel= -0.29
  t=  74/240  realized: pos= 24.3mm  rot=  3.5°  frac_term=0.00  rate=0.31 t/s  ETA=528s
           plan: rew= -18.51  best_sample=  -3.55  (MPPI improv +0.45)
           per-iter rew_max:  -4.00  -4.01  -3.88  -3.85  -3.79  -3.67  -3.60  -3.55
           per-iter plan_rew: -4.02  -3.99  -4.01  -25.69  -3.89  -3.75  -3.60  -18.51
           plan terms (forecast): cube_pos= -0.47  cube_quat=-11.00  finger_qpos= -0.18  contact= -4.67  wrist_delta= -0.19  qvel= -0.35  cube_angvel= -1.65
  t=  75/240  realized: pos= 25.8mm  rot=  3.8°  frac_term=0.00  rate=0.31 t/s  ETA=525s
           plan: rew=  -3.53  best_sample=  -3.52  (MPPI improv +0.50)
           per-iter rew_max:  -4.02  -4.01  -3.79  -3.66  -3.66  -3.64  -3.56  -3.52
           per-iter plan_rew: -4.02  -3.94  -3.83  -3.66  -3.66  -3.86  -3.55  -3.53
           plan terms (forecast): cube_pos= -0.14  cube_quat= -0.64  finger_qpos= -0.20  contact= -1.78  wrist_delta= -0.20  qvel= -0.30  cube_angvel= -0.28
  t=  76/240  realized: pos= 24.2mm  rot=  1.9°  frac_term=0.00  rate=0.31 t/s  ETA=522s
           plan: rew=  -3.66  best_sample=  -3.63  (MPPI improv +0.73)
           per-iter rew_max:  -4.35  -4.30  -4.13  -4.04  -3.76  -3.73  -3.68  -3.63
           per-iter plan_rew: -4.31  -4.24  -4.21  -3.96  -3.75  -3.92  -3.71  -3.66
           plan terms (forecast): cube_pos= -0.13  cube_quat= -0.69  finger_qpos= -0.20  contact= -1.85  wrist_delta= -0.19  qvel= -0.31  cube_angvel= -0.30
  t=  77/240  realized: pos= 22.6mm  rot=  4.6°  frac_term=0.00  rate=0.31 t/s  ETA=520s
           plan: rew=  -3.49  best_sample=  -3.50  (MPPI improv +0.95)
           per-iter rew_max:  -4.45  -4.11  -3.97  -3.80  -3.66  -3.61  -3.56  -3.50
           per-iter plan_rew: -4.28  -4.01  -3.91  -3.84  -3.63  -3.70  -3.62  -3.49
           plan terms (forecast): cube_pos= -0.11  cube_quat= -0.55  finger_qpos= -0.20  contact= -1.89  wrist_delta= -0.18  qvel= -0.29  cube_angvel= -0.26
  t=  78/240  realized: pos= 19.9mm  rot=  5.2°  frac_term=0.00  rate=0.31 t/s  ETA=517s
           plan: rew=  -3.39  best_sample=  -3.40  (MPPI improv +0.68)
           per-iter rew_max:  -4.08  -3.75  -3.71  -3.65  -3.56  -3.51  -3.45  -3.40
           per-iter plan_rew: -3.95  -3.72  -3.66  -3.63  -3.53  -3.50  -3.44  -3.39
           plan terms (forecast): cube_pos= -0.10  cube_quat= -0.52  finger_qpos= -0.23  contact= -1.90  wrist_delta= -0.19  qvel= -0.26  cube_angvel= -0.19
  t=  79/240  realized: pos= 17.9mm  rot=  1.9°  frac_term=0.00  rate=0.31 t/s  ETA=515s
           plan: rew=  -3.46  best_sample=  -3.46  (MPPI improv +0.61)
           per-iter rew_max:  -4.06  -4.00  -3.84  -3.67  -3.60  -3.51  -3.49  -3.46
           per-iter plan_rew: -4.19  -3.98  -3.79  -3.66  -3.61  -3.50  -3.48  -3.46
           plan terms (forecast): cube_pos= -0.09  cube_quat= -0.56  finger_qpos= -0.23  contact= -1.95  wrist_delta= -0.16  qvel= -0.29  cube_angvel= -0.19
  t=  80/240  realized: pos= 13.9mm  rot=  2.8°  frac_term=0.00  rate=0.31 t/s  ETA=512s
           plan: rew=  -3.45  best_sample=  -3.43  (MPPI improv +0.45)
           per-iter rew_max:  -3.88  -3.88  -3.90  -3.76  -3.53  -3.49  -3.48  -3.43
           per-iter plan_rew: -3.88  -3.92  -3.87  -3.78  -3.51  -3.50  -3.45  -3.45
           plan terms (forecast): cube_pos= -0.09  cube_quat= -0.55  finger_qpos= -0.23  contact= -1.94  wrist_delta= -0.15  qvel= -0.28  cube_angvel= -0.21
  t=  81/240  realized: pos=  9.0mm  rot=  3.8°  frac_term=0.00  rate=0.31 t/s  ETA=509s
           plan: rew=  -3.49  best_sample=  -3.51  (MPPI improv +0.73)
           per-iter rew_max:  -4.24  -3.77  -3.75  -3.75  -3.70  -3.56  -3.53  -3.51
           per-iter plan_rew: -3.99  -3.75  -3.75  -3.76  -3.65  -3.59  -3.55  -3.49
           plan terms (forecast): cube_pos= -0.08  cube_quat= -0.54  finger_qpos= -0.25  contact= -1.97  wrist_delta= -0.20  qvel= -0.25  cube_angvel= -0.20
  t=  82/240  realized: pos=  8.9mm  rot=  3.3°  frac_term=0.00  rate=0.31 t/s  ETA=507s
           plan: rew=  -3.59  best_sample=  -3.60  (MPPI improv +0.47)
           per-iter rew_max:  -4.07  -4.04  -3.86  -3.82  -3.73  -3.69  -3.63  -3.60
           per-iter plan_rew: -4.07  -4.07  -3.82  -3.80  -3.72  -3.67  -3.70  -3.59
           plan terms (forecast): cube_pos= -0.10  cube_quat= -0.44  finger_qpos= -0.25  contact= -2.10  wrist_delta= -0.22  qvel= -0.28  cube_angvel= -0.20
  t=  83/240  realized: pos=  8.8mm  rot=  1.5°  frac_term=0.00  rate=0.31 t/s  ETA=505s
           plan: rew=  -3.36  best_sample=  -3.36  (MPPI improv +0.78)
           per-iter rew_max:  -4.14  -3.83  -3.62  -3.55  -3.51  -3.44  -3.40  -3.36
           per-iter plan_rew: -3.93  -3.68  -3.56  -3.52  -3.47  -3.41  -3.39  -3.36
           plan terms (forecast): cube_pos= -0.12  cube_quat= -0.33  finger_qpos= -0.26  contact= -1.95  wrist_delta= -0.24  qvel= -0.29  cube_angvel= -0.17
  t=  84/240  realized: pos=  9.5mm  rot=  1.2°  frac_term=0.00  rate=0.31 t/s  ETA=502s
           plan: rew=  -3.44  best_sample=  -3.43  (MPPI improv +0.59)
           per-iter rew_max:  -4.02  -3.77  -3.70  -3.61  -3.53  -3.52  -3.46  -3.43
           per-iter plan_rew: -3.99  -3.78  -3.69  -3.53  -3.53  -3.53  -3.48  -3.44
           plan terms (forecast): cube_pos= -0.11  cube_quat= -0.40  finger_qpos= -0.26  contact= -1.98  wrist_delta= -0.24  qvel= -0.27  cube_angvel= -0.18
  t=  85/240  realized: pos=  7.0mm  rot=  2.1°  frac_term=0.00  rate=0.31 t/s  ETA=499s
           plan: rew=  -3.43  best_sample=  -3.42  (MPPI improv +0.63)
           per-iter rew_max:  -4.05  -3.79  -3.70  -3.62  -3.53  -3.49  -3.42  -3.42
           per-iter plan_rew: -4.86  -3.72  -3.64  -3.60  -3.52  -3.49  -3.42  -3.43
           plan terms (forecast): cube_pos= -0.12  cube_quat= -0.41  finger_qpos= -0.29  contact= -1.89  wrist_delta= -0.27  qvel= -0.30  cube_angvel= -0.16
  t=  86/240  realized: pos=  8.2mm  rot=  2.6°  frac_term=0.00  rate=0.31 t/s  ETA=497s
           plan: rew=  -3.40  best_sample=  -3.39  (MPPI improv +0.61)
           per-iter rew_max:  -4.00  -3.78  -3.67  -3.53  -3.50  -3.47  -3.42  -3.39
           per-iter plan_rew: -3.83  -3.70  -3.54  -3.51  -3.51  -3.46  -3.42  -3.40
           plan terms (forecast): cube_pos= -0.11  cube_quat= -0.46  finger_qpos= -0.28  contact= -1.81  wrist_delta= -0.21  qvel= -0.32  cube_angvel= -0.20
  t=  87/240  realized: pos=  9.8mm  rot=  2.2°  frac_term=0.00  rate=0.31 t/s  ETA=494s
           plan: rew=  -3.66  best_sample=  -3.54  (MPPI improv +0.41)
           per-iter rew_max:  -3.95  -3.94  -3.91  -3.86  -3.74  -3.64  -3.60  -3.54
           per-iter plan_rew: -3.94  -3.93  -3.91  -3.81  -3.78  -3.64  -3.57  -3.66
           plan terms (forecast): cube_pos= -0.12  cube_quat= -0.71  finger_qpos= -0.27  contact= -1.78  wrist_delta= -0.19  qvel= -0.35  cube_angvel= -0.24
  t=  88/240  realized: pos=  7.9mm  rot=  0.9°  frac_term=0.00  rate=0.31 t/s  ETA=491s
           plan: rew=  -3.45  best_sample=  -3.45  (MPPI improv +0.66)
           per-iter rew_max:  -4.11  -4.02  -3.83  -3.71  -3.66  -3.57  -3.49  -3.45
           per-iter plan_rew: -4.19  -3.92  -3.81  -3.66  -3.68  -3.56  -3.49  -3.45
           plan terms (forecast): cube_pos= -0.13  cube_quat= -0.56  finger_qpos= -0.26  contact= -1.78  wrist_delta= -0.21  qvel= -0.31  cube_angvel= -0.20
  t=  89/240  realized: pos=  7.4mm  rot=  0.8°  frac_term=0.00  rate=0.31 t/s  ETA=489s
           plan: rew=  -3.15  best_sample=  -3.12  (MPPI improv +0.46)
           per-iter rew_max:  -3.58  -3.58  -3.52  -3.35  -3.29  -3.21  -3.16  -3.12
           per-iter plan_rew: -3.58  -3.63  -3.53  -3.38  -3.28  -3.20  -3.16  -3.15
           plan terms (forecast): cube_pos= -0.12  cube_quat= -0.36  finger_qpos= -0.26  contact= -1.71  wrist_delta= -0.19  qvel= -0.36  cube_angvel= -0.13
  t=  90/240  realized: pos=  6.5mm  rot=  0.7°  frac_term=0.00  rate=0.31 t/s  ETA=486s
           plan: rew=  -3.26  best_sample=  -3.02  (MPPI improv +0.21)
           per-iter rew_max:  -3.23  -3.24  -3.16  -3.23  -3.25  -3.20  -3.04  -3.02
           per-iter plan_rew: -3.24  -3.50  -3.23  -3.26  -3.20  -3.17  -3.06  -3.26
           plan terms (forecast): cube_pos= -0.14  cube_quat= -0.47  finger_qpos= -0.25  contact= -1.69  wrist_delta= -0.18  qvel= -0.34  cube_angvel= -0.19
  t=  91/240  realized: pos=  4.9mm  rot=  1.9°  frac_term=0.00  rate=0.31 t/s  ETA=483s
           plan: rew=  -2.96  best_sample=  -2.95  (MPPI improv +0.57)
           per-iter rew_max:  -3.52  -3.46  -3.18  -3.14  -3.10  -3.02  -2.96  -2.95
           per-iter plan_rew: -3.64  -7.77  -3.15  -3.18  -3.07  -3.02  -3.02  -2.96
           plan terms (forecast): cube_pos= -0.17  cube_quat= -0.34  finger_qpos= -0.25  contact= -1.62  wrist_delta= -0.12  qvel= -0.34  cube_angvel= -0.11
  t=  92/240  realized: pos=  5.3mm  rot=  2.9°  frac_term=0.00  rate=0.31 t/s  ETA=481s
           plan: rew=  -3.01  best_sample=  -3.00  (MPPI improv +0.32)
           per-iter rew_max:  -3.32  -3.26  -3.19  -3.13  -3.07  -3.05  -3.03  -3.00
           per-iter plan_rew: -3.26  -3.26  -3.20  -3.12  -3.08  -3.03  -3.03  -3.01
           plan terms (forecast): cube_pos= -0.17  cube_quat= -0.36  finger_qpos= -0.25  contact= -1.60  wrist_delta= -0.13  qvel= -0.36  cube_angvel= -0.15
  t=  93/240  realized: pos=  5.1mm  rot=  2.3°  frac_term=0.00  rate=0.31 t/s  ETA=479s
           plan: rew=  -2.99  best_sample=  -2.96  (MPPI improv +0.32)
           per-iter rew_max:  -3.27  -3.15  -3.17  -3.14  -3.08  -3.07  -3.01  -2.96
           per-iter plan_rew: -3.27  -3.17  -3.18  -3.13  -3.18  -3.13  -3.11  -2.99
           plan terms (forecast): cube_pos= -0.19  cube_quat= -0.32  finger_qpos= -0.24  contact= -1.57  wrist_delta= -0.17  qvel= -0.37  cube_angvel= -0.13
  t=  94/240  realized: pos=  5.9mm  rot=  1.3°  frac_term=0.00  rate=0.30 t/s  ETA=476s
           plan: rew=  -2.86  best_sample=  -2.85  (MPPI improv +0.49)
           per-iter rew_max:  -3.34  -3.31  -3.19  -3.07  -3.01  -2.92  -2.90  -2.85
           per-iter plan_rew: -3.34  -3.19  -3.21  -3.01  -2.98  -2.95  -2.87  -2.86
           plan terms (forecast): cube_pos= -0.18  cube_quat= -0.29  finger_qpos= -0.24  contact= -1.42  wrist_delta= -0.16  qvel= -0.41  cube_angvel= -0.15
  t=  95/240  realized: pos=  9.6mm  rot=  1.3°  frac_term=0.00  rate=0.30 t/s  ETA=474s
           plan: rew=  -3.07  best_sample=  -2.85  (MPPI improv +0.42)
           per-iter rew_max:  -3.27  -2.96  -2.95  -3.00  -2.94  -2.97  -2.85  -2.85
           per-iter plan_rew: -2.96  -3.05  -3.01  -2.98  -3.18  -2.97  -3.09  -3.07
           plan terms (forecast): cube_pos= -0.19  cube_quat= -0.55  finger_qpos= -0.22  contact= -1.35  wrist_delta= -0.15  qvel= -0.41  cube_angvel= -0.20
  t=  96/240  realized: pos= 11.5mm  rot=  0.7°  frac_term=0.00  rate=0.30 t/s  ETA=472s
           plan: rew=  -2.83  best_sample=  -2.75  (MPPI improv +0.49)
           per-iter rew_max:  -3.24  -3.08  -3.06  -2.91  -2.88  -2.77  -2.80  -2.75
           per-iter plan_rew: -4.16  -4.43  -2.98  -3.43  -2.84  -3.12  -2.85  -2.83
           plan terms (forecast): cube_pos= -0.17  cube_quat= -0.40  finger_qpos= -0.22  contact= -1.30  wrist_delta= -0.19  qvel= -0.39  cube_angvel= -0.16
  t=  97/240  realized: pos= 14.2mm  rot=  1.0°  frac_term=0.00  rate=0.30 t/s  ETA=469s
           plan: rew=  -2.78  best_sample=  -2.76  (MPPI improv +0.35)
           per-iter rew_max:  -3.11  -3.21  -3.11  -3.08  -3.00  -2.83  -2.79  -2.76
           per-iter plan_rew: -5.10  -3.18  -3.18  -9.66  -2.97  -3.05  -2.89  -2.78
           plan terms (forecast): cube_pos= -0.20  cube_quat= -0.40  finger_qpos= -0.21  contact= -1.22  wrist_delta= -0.19  qvel= -0.38  cube_angvel= -0.18
  t=  98/240  realized: pos= 17.3mm  rot=  2.8°  frac_term=0.00  rate=0.30 t/s  ETA=466s
           plan: rew=  -2.83  best_sample=  -2.80  (MPPI improv +0.66)
           per-iter rew_max:  -3.46  -3.23  -3.08  -3.01  -3.02  -2.90  -2.80  -2.80
           per-iter plan_rew: -3.75  -3.25  -3.10  -3.82  -3.01  -2.99  -2.94  -2.83
           plan terms (forecast): cube_pos= -0.19  cube_quat= -0.48  finger_qpos= -0.22  contact= -1.13  wrist_delta= -0.18  qvel= -0.42  cube_angvel= -0.21
  t=  99/240  realized: pos= 22.1mm  rot=  1.9°  frac_term=0.00  rate=0.30 t/s  ETA=464s
           plan: rew=  -2.78  best_sample=  -2.73  (MPPI improv +0.65)
           per-iter rew_max:  -3.38  -2.98  -3.00  -2.96  -2.87  -2.84  -2.78  -2.73
           per-iter plan_rew: -2.98  -4.17  -3.17  -2.90  -2.87  -2.82  -2.76  -2.78
           plan terms (forecast): cube_pos= -0.18  cube_quat= -0.52  finger_qpos= -0.21  contact= -1.04  wrist_delta= -0.17  qvel= -0.43  cube_angvel= -0.22
  t= 100/240  realized: pos= 21.6mm  rot=  0.9°  frac_term=0.00  rate=0.30 t/s  ETA=461s
           plan: rew=  -2.98  best_sample=  -2.78  (MPPI improv +0.16)
           per-iter rew_max:  -2.94  -3.26  -3.15  -2.83  -2.85  -2.84  -2.84  -2.78
           per-iter plan_rew: -3.26  -3.24  -3.27  -3.26  -2.88  -3.59  -2.84  -2.98
           plan terms (forecast): cube_pos= -0.17  cube_quat= -0.76  finger_qpos= -0.20  contact= -1.00  wrist_delta= -0.22  qvel= -0.41  cube_angvel= -0.23
  t= 101/240  realized: pos= 19.5mm  rot=  2.2°  frac_term=0.00  rate=0.30 t/s  ETA=458s
           plan: rew=  -2.53  best_sample=  -2.51  (MPPI improv +0.41)
           per-iter rew_max:  -2.92  -2.82  -2.90  -2.66  -2.64  -2.59  -2.53  -2.51
           per-iter plan_rew: -9.30  -7.62  -5.07  -2.79  -3.10  -3.01  -2.57  -2.53
           plan terms (forecast): cube_pos= -0.18  cube_quat= -0.44  finger_qpos= -0.20  contact= -0.99  wrist_delta= -0.20  qvel= -0.35  cube_angvel= -0.18
  t= 102/240  realized: pos= 17.6mm  rot=  2.2°  frac_term=0.00  rate=0.30 t/s  ETA=454s
           plan: rew=  -2.74  best_sample=  -2.51  (MPPI improv +0.20)
           per-iter rew_max:  -2.72  -2.76  -2.58  -2.67  -2.65  -2.59  -2.52  -2.51
           per-iter plan_rew: -2.84  -2.59  -3.18  -3.55  -2.92  -2.73  -2.73  -2.74
           plan terms (forecast): cube_pos= -0.19  cube_quat= -0.83  finger_qpos= -0.17  contact= -0.79  wrist_delta= -0.22  qvel= -0.32  cube_angvel= -0.21
  t= 103/240  realized: pos= 16.9mm  rot=  0.7°  frac_term=0.00  rate=0.30 t/s  ETA=452s
           plan: rew=  -9.41  best_sample=  -2.17  (MPPI improv +0.46)
           per-iter rew_max:  -2.63  -2.69  -2.35  -2.36  -2.25  -2.19  -2.19  -2.17
           per-iter plan_rew: -2.80  -2.37  -2.36  -3.39  -3.50  -3.00  -2.26  -9.41
           plan terms (forecast): cube_pos= -0.21  cube_quat= -7.03  finger_qpos= -0.18  contact= -0.98  wrist_delta= -0.20  qvel= -0.31  cube_angvel= -0.50
  t= 104/240  realized: pos= 18.9mm  rot=  1.7°  frac_term=0.00  rate=0.30 t/s  ETA=448s
           plan: rew=  -2.07  best_sample=  -1.99  (MPPI improv +0.73)
           per-iter rew_max:  -2.72  -2.33  -2.30  -2.33  -2.28  -2.09  -2.03  -1.99
           per-iter plan_rew: -2.33  -2.31  -2.50  -2.51  -2.55  -9.29  -10.45  -2.07
           plan terms (forecast): cube_pos= -0.14  cube_quat= -0.40  finger_qpos= -0.18  contact= -0.69  wrist_delta= -0.21  qvel= -0.28  cube_angvel= -0.16
  t= 105/240  realized: pos= 20.8mm  rot=  2.2°  frac_term=0.00  rate=0.30 t/s  ETA=445s
           plan: rew=  -2.25  best_sample=  -1.96  (MPPI improv +0.07)
           per-iter rew_max:  -2.03  -2.10  -2.03  -2.10  -2.12  -2.07  -2.00  -1.96
           per-iter plan_rew: -2.10  -2.03  -2.10  -10.05  -2.25  -2.08  -1.99  -2.25
           plan terms (forecast): cube_pos= -0.13  cube_quat= -0.62  finger_qpos= -0.17  contact= -0.68  wrist_delta= -0.19  qvel= -0.32  cube_angvel= -0.14
  t= 106/240  realized: pos= 20.9mm  rot=  1.4°  frac_term=0.00  rate=0.30 t/s  ETA=442s
           plan: rew=  -1.97  best_sample=  -1.92  (MPPI improv +0.12)
           per-iter rew_max:  -2.04  -2.09  -2.06  -2.10  -2.07  -2.00  -1.97  -1.92
           per-iter plan_rew: -2.20  -2.18  -10.63  -10.29  -2.18  -2.02  -9.89  -1.97
           plan terms (forecast): cube_pos= -0.13  cube_quat= -0.39  finger_qpos= -0.17  contact= -0.63  wrist_delta= -0.19  qvel= -0.32  cube_angvel= -0.14
  t= 107/240  realized: pos= 21.6mm  rot=  2.9°  frac_term=0.00  rate=0.30 t/s  ETA=439s
           plan: rew=  -2.04  best_sample=  -1.89  (MPPI improv +0.31)
           per-iter rew_max:  -2.20  -2.29  -2.31  -2.05  -2.02  -1.97  -1.91  -1.89
           per-iter plan_rew: -2.73  -2.54  -2.18  -2.06  -2.01  -2.07  -2.06  -2.04
           plan terms (forecast): cube_pos= -0.13  cube_quat= -0.52  finger_qpos= -0.16  contact= -0.61  wrist_delta= -0.17  qvel= -0.30  cube_angvel= -0.14
  t= 108/240  realized: pos= 21.5mm  rot=  4.1°  frac_term=0.00  rate=0.30 t/s  ETA=435s
           plan: rew=  -1.81  best_sample=  -1.79  (MPPI improv +0.80)
           per-iter rew_max:  -2.59  -2.08  -2.04  -1.92  -1.87  -1.84  -1.81  -1.79
           per-iter plan_rew: -2.12  -2.12  -2.12  -1.89  -1.96  -1.86  -1.94  -1.81
           plan terms (forecast): cube_pos= -0.13  cube_quat= -0.36  finger_qpos= -0.15  contact= -0.63  wrist_delta= -0.12  qvel= -0.29  cube_angvel= -0.13
  t= 109/240  realized: pos= 19.4mm  rot=  5.1°  frac_term=0.00  rate=0.30 t/s  ETA=432s
           plan: rew=  -2.22  best_sample=  -1.82  (MPPI improv +0.40)
           per-iter rew_max:  -2.21  -2.03  -2.02  -1.95  -1.94  -1.88  -1.84  -1.82
           per-iter plan_rew: -2.01  -2.11  -1.95  -2.06  -1.97  -2.02  -1.90  -2.22
           plan terms (forecast): cube_pos= -0.13  cube_quat= -0.75  finger_qpos= -0.16  contact= -0.62  wrist_delta= -0.13  qvel= -0.29  cube_angvel= -0.14
  t= 110/240  realized: pos= 16.6mm  rot=  2.4°  frac_term=0.00  rate=0.30 t/s  ETA=429s
           plan: rew=  -1.87  best_sample=  -1.87  (MPPI improv +0.85)
           per-iter rew_max:  -2.72  -2.40  -2.25  -2.08  -2.02  -1.93  -1.90  -1.87
           per-iter plan_rew: -2.73  -2.42  -2.21  -2.06  -1.99  -1.93  -1.89  -1.87
           plan terms (forecast): cube_pos= -0.13  cube_quat= -0.35  finger_qpos= -0.18  contact= -0.64  wrist_delta= -0.18  qvel= -0.28  cube_angvel= -0.11
  t= 111/240  realized: pos= 14.6mm  rot=  2.0°  frac_term=0.00  rate=0.30 t/s  ETA=425s
           plan: rew=  -1.89  best_sample=  -1.86  (MPPI improv +0.23)
           per-iter rew_max:  -2.09  -2.09  -2.01  -2.01  -1.93  -1.90  -1.88  -1.86
           per-iter plan_rew: -2.10  -2.02  -2.01  -2.01  -1.94  -1.89  -1.87  -1.89
           plan terms (forecast): cube_pos= -0.12  cube_quat= -0.36  finger_qpos= -0.18  contact= -0.63  wrist_delta= -0.16  qvel= -0.28  cube_angvel= -0.16
  t= 112/240  realized: pos= 14.8mm  rot=  2.4°  frac_term=0.00  rate=0.30 t/s  ETA=421s
           plan: rew=  -1.96  best_sample=  -1.89  (MPPI improv +0.26)
           per-iter rew_max:  -2.16  -2.10  -2.09  -2.01  -2.02  -1.94  -1.92  -1.89
           per-iter plan_rew: -2.17  -2.21  -2.01  -2.20  -2.00  -2.19  -1.97  -1.96
           plan terms (forecast): cube_pos= -0.13  cube_quat= -0.40  finger_qpos= -0.18  contact= -0.62  wrist_delta= -0.18  qvel= -0.30  cube_angvel= -0.15
  t= 113/240  realized: pos= 13.0mm  rot=  1.9°  frac_term=0.00  rate=0.30 t/s  ETA=417s
           plan: rew=  -1.76  best_sample=  -1.77  (MPPI improv +0.40)
           per-iter rew_max:  -2.16  -2.13  -2.05  -1.97  -1.90  -1.84  -1.80  -1.77
           per-iter plan_rew: -2.30  -2.10  -2.06  -1.94  -1.88  -1.83  -1.81  -1.76
           plan terms (forecast): cube_pos= -0.13  cube_quat= -0.28  finger_qpos= -0.19  contact= -0.62  wrist_delta= -0.15  qvel= -0.29  cube_angvel= -0.10
  t= 114/240  realized: pos= 10.2mm  rot=  1.1°  frac_term=0.00  rate=0.30 t/s  ETA=414s
           plan: rew=  -1.87  best_sample=  -1.86  (MPPI improv +0.32)
           per-iter rew_max:  -2.18  -2.18  -2.17  -2.06  -1.97  -1.92  -1.90  -1.86
           per-iter plan_rew: -2.18  -2.18  -2.14  -2.06  -1.96  -1.91  -1.89  -1.87
           plan terms (forecast): cube_pos= -0.12  cube_quat= -0.34  finger_qpos= -0.19  contact= -0.61  wrist_delta= -0.14  qvel= -0.30  cube_angvel= -0.17
  t= 115/240  realized: pos=  9.9mm  rot=  0.4°  frac_term=0.00  rate=0.30 t/s  ETA=410s
           plan: rew=  -1.84  best_sample=  -1.84  (MPPI improv +0.55)
           per-iter rew_max:  -2.39  -2.22  -2.09  -2.00  -1.96  -1.92  -1.87  -1.84
           per-iter plan_rew: -2.24  -2.10  -2.03  -2.00  -1.95  -1.91  -1.87  -1.84
           plan terms (forecast): cube_pos= -0.12  cube_quat= -0.35  finger_qpos= -0.20  contact= -0.60  wrist_delta= -0.12  qvel= -0.27  cube_angvel= -0.18
  t= 116/240  realized: pos= 11.3mm  rot=  1.2°  frac_term=0.00  rate=0.30 t/s  ETA=407s
           plan: rew=  -1.85  best_sample=  -1.84  (MPPI improv +0.64)
           per-iter rew_max:  -2.48  -2.31  -2.11  -2.03  -1.99  -1.93  -1.90  -1.84
           per-iter plan_rew: -2.49  -2.18  -2.07  -2.00  -1.94  -1.93  -1.90  -1.85
           plan terms (forecast): cube_pos= -0.11  cube_quat= -0.38  finger_qpos= -0.20  contact= -0.61  wrist_delta= -0.14  qvel= -0.28  cube_angvel= -0.14
  t= 117/240  realized: pos= 15.3mm  rot=  1.4°  frac_term=0.00  rate=0.30 t/s  ETA=404s
           plan: rew=  -1.99  best_sample=  -2.01  (MPPI improv +0.44)
           per-iter rew_max:  -2.44  -2.44  -2.32  -2.24  -2.12  -2.08  -2.04  -2.01
           per-iter plan_rew: -2.48  -2.36  -2.28  -2.20  -2.08  -2.06  -2.07  -1.99
           plan terms (forecast): cube_pos= -0.10  cube_quat= -0.41  finger_qpos= -0.19  contact= -0.63  wrist_delta= -0.14  qvel= -0.29  cube_angvel= -0.23
  t= 118/240  realized: pos= 19.0mm  rot=  0.7°  frac_term=0.00  rate=0.30 t/s  ETA=400s
           plan: rew=  -2.31  best_sample=  -2.08  (MPPI improv +0.42)
           per-iter rew_max:  -2.50  -2.42  -2.41  -2.23  -2.20  -2.10  -2.09  -2.08
           per-iter plan_rew: -2.42  -2.63  -2.40  -2.83  -2.38  -2.24  -2.09  -2.31
           plan terms (forecast): cube_pos= -0.12  cube_quat= -0.65  finger_qpos= -0.17  contact= -0.66  wrist_delta= -0.18  qvel= -0.30  cube_angvel= -0.24
  t= 119/240  realized: pos= 20.8mm  rot=  1.4°  frac_term=0.00  rate=0.30 t/s  ETA=397s
           plan: rew=  -2.09  best_sample=  -2.09  (MPPI improv +0.95)
           per-iter rew_max:  -3.03  -2.55  -2.45  -2.33  -2.25  -2.13  -2.10  -2.09
           per-iter plan_rew: -2.87  -2.54  -2.39  -2.31  -2.17  -2.10  -2.11  -2.09
           plan terms (forecast): cube_pos= -0.12  cube_quat= -0.43  finger_qpos= -0.19  contact= -0.71  wrist_delta= -0.17  qvel= -0.29  cube_angvel= -0.19
  t= 120/240  realized: pos= 20.8mm  rot=  2.9°  frac_term=0.00  rate=0.30 t/s  ETA=393s
           plan: rew=  -2.23  best_sample=  -2.24  (MPPI improv +0.48)
           per-iter rew_max:  -2.72  -2.58  -2.55  -2.40  -2.38  -2.29  -2.24  -2.24
           per-iter plan_rew: -2.69  -2.78  -2.63  -2.42  -2.45  -2.36  -2.69  -2.23
           plan terms (forecast): cube_pos= -0.11  cube_quat= -0.51  finger_qpos= -0.18  contact= -0.71  wrist_delta= -0.16  qvel= -0.35  cube_angvel= -0.21
  t= 121/240  realized: pos= 18.2mm  rot=  1.6°  frac_term=0.00  rate=0.30 t/s  ETA=390s
           plan: rew=  -2.79  best_sample=  -2.24  (MPPI improv +0.53)
           per-iter rew_max:  -2.78  -2.70  -2.51  -2.39  -2.32  -2.30  -2.28  -2.24
           per-iter plan_rew: -3.02  -2.79  -2.51  -2.40  -2.47  -2.47  -2.28  -2.79
           plan terms (forecast): cube_pos= -0.11  cube_quat= -0.91  finger_qpos= -0.20  contact= -0.78  wrist_delta= -0.17  qvel= -0.32  cube_angvel= -0.30
  t= 122/240  realized: pos= 14.9mm  rot=  2.2°  frac_term=0.00  rate=0.30 t/s  ETA=387s
           plan: rew=  -2.61  best_sample=  -2.42  (MPPI improv +0.52)
           per-iter rew_max:  -2.94  -2.79  -2.60  -2.58  -2.50  -2.44  -2.42  -2.42
           per-iter plan_rew: -2.89  -2.98  -2.71  -2.66  -2.53  -2.44  -2.49  -2.61
           plan terms (forecast): cube_pos= -0.13  cube_quat= -0.67  finger_qpos= -0.21  contact= -0.84  wrist_delta= -0.15  qvel= -0.32  cube_angvel= -0.28
  t= 123/240  realized: pos= 11.6mm  rot=  1.3°  frac_term=0.00  rate=0.30 t/s  ETA=384s
           plan: rew=  -2.44  best_sample=  -2.37  (MPPI improv +0.44)
           per-iter rew_max:  -2.80  -2.80  -2.57  -2.54  -2.51  -2.41  -2.38  -2.37
           per-iter plan_rew: -2.84  -2.78  -2.56  -2.60  -2.48  -2.52  -2.65  -2.44
           plan terms (forecast): cube_pos= -0.13  cube_quat= -0.55  finger_qpos= -0.22  contact= -0.87  wrist_delta= -0.13  qvel= -0.32  cube_angvel= -0.22
  t= 124/240  realized: pos= 12.2mm  rot=  2.4°  frac_term=0.00  rate=0.30 t/s  ETA=380s
           plan: rew=  -2.44  best_sample=  -2.44  (MPPI improv +0.78)
           per-iter rew_max:  -3.22  -3.01  -2.83  -2.63  -2.60  -2.52  -2.47  -2.44
           per-iter plan_rew: -3.68  -4.69  -2.74  -2.65  -2.62  -2.60  -2.59  -2.44
           plan terms (forecast): cube_pos= -0.12  cube_quat= -0.61  finger_qpos= -0.22  contact= -0.85  wrist_delta= -0.14  qvel= -0.30  cube_angvel= -0.21
  t= 125/240  realized: pos= 14.4mm  rot=  2.8°  frac_term=0.00  rate=0.30 t/s  ETA=377s
           plan: rew=  -2.68  best_sample=  -2.67  (MPPI improv +0.49)
           per-iter rew_max:  -3.15  -3.06  -2.98  -2.91  -2.88  -2.75  -2.69  -2.67
           per-iter plan_rew: -3.08  -2.98  -2.92  -2.92  -2.85  -4.07  -2.82  -2.68
           plan terms (forecast): cube_pos= -0.12  cube_quat= -0.62  finger_qpos= -0.22  contact= -1.02  wrist_delta= -0.14  qvel= -0.32  cube_angvel= -0.25
  t= 126/240  realized: pos= 14.9mm  rot=  3.6°  frac_term=0.00  rate=0.30 t/s  ETA=374s
           plan: rew=  -3.05  best_sample=  -2.88  (MPPI improv +0.60)
           per-iter rew_max:  -3.48  -3.17  -3.15  -3.12  -2.96  -2.94  -2.91  -2.88
           per-iter plan_rew: -3.18  -3.35  -3.25  -3.15  -2.99  -2.96  -2.93  -3.05
           plan terms (forecast): cube_pos= -0.12  cube_quat= -0.80  finger_qpos= -0.22  contact= -1.13  wrist_delta= -0.13  qvel= -0.32  cube_angvel= -0.32
  t= 127/240  realized: pos= 16.7mm  rot=  2.6°  frac_term=0.00  rate=0.30 t/s  ETA=371s
           plan: rew=  -2.76  best_sample=  -2.76  (MPPI improv +0.90)
           per-iter rew_max:  -3.66  -3.06  -3.14  -2.98  -2.95  -2.89  -2.82  -2.76
           per-iter plan_rew: -4.45  -3.14  -3.16  -2.97  -2.94  -2.92  -2.84  -2.76
           plan terms (forecast): cube_pos= -0.14  cube_quat= -0.52  finger_qpos= -0.23  contact= -1.16  wrist_delta= -0.19  qvel= -0.31  cube_angvel= -0.21
  t= 128/240  realized: pos= 19.2mm  rot=  0.7°  frac_term=0.00  rate=0.30 t/s  ETA=368s
           plan: rew=  -2.80  best_sample=  -2.79  (MPPI improv +0.53)
           per-iter rew_max:  -3.32  -3.18  -3.06  -2.87  -2.85  -2.81  -2.81  -2.79
           per-iter plan_rew: -3.18  -3.09  -3.16  -2.86  -2.87  -2.81  -2.81  -2.80
           plan terms (forecast): cube_pos= -0.15  cube_quat= -0.45  finger_qpos= -0.24  contact= -1.22  wrist_delta= -0.24  qvel= -0.31  cube_angvel= -0.19
  t= 129/240  realized: pos= 19.1mm  rot=  1.4°  frac_term=0.00  rate=0.30 t/s  ETA=364s
           plan: rew=  -2.99  best_sample=  -2.93  (MPPI improv +0.52)
           per-iter rew_max:  -3.45  -3.34  -3.34  -3.16  -3.06  -3.03  -2.97  -2.93
           per-iter plan_rew: -3.44  -3.34  -3.36  -3.15  -3.04  -3.52  -3.25  -2.99
           plan terms (forecast): cube_pos= -0.14  cube_quat= -0.55  finger_qpos= -0.23  contact= -1.28  wrist_delta= -0.25  qvel= -0.32  cube_angvel= -0.21
  t= 130/240  realized: pos= 18.5mm  rot=  2.4°  frac_term=0.00  rate=0.30 t/s  ETA=361s
           plan: rew=  -2.84  best_sample=  -2.83  (MPPI improv +0.63)
           per-iter rew_max:  -3.46  -3.38  -3.26  -3.12  -3.01  -2.93  -2.88  -2.83
           per-iter plan_rew: -3.44  -3.34  -3.21  -3.09  -3.00  -2.94  -2.88  -2.84
           plan terms (forecast): cube_pos= -0.12  cube_quat= -0.43  finger_qpos= -0.24  contact= -1.35  wrist_delta= -0.19  qvel= -0.32  cube_angvel= -0.18
  t= 131/240  realized: pos= 19.4mm  rot=  1.1°  frac_term=0.00  rate=0.30 t/s  ETA=358s
           plan: rew=  -2.77  best_sample=  -2.75  (MPPI improv +0.70)
           per-iter rew_max:  -3.44  -3.22  -3.08  -2.97  -2.86  -2.80  -2.81  -2.75
           per-iter plan_rew: -3.35  -3.18  -2.99  -2.91  -3.61  -3.45  -2.98  -2.77
           plan terms (forecast): cube_pos= -0.12  cube_quat= -0.45  finger_qpos= -0.25  contact= -1.30  wrist_delta= -0.15  qvel= -0.33  cube_angvel= -0.17
  t= 132/240  realized: pos= 19.9mm  rot=  4.1°  frac_term=0.00  rate=0.30 t/s  ETA=355s
           plan: rew=  -2.81  best_sample=  -2.82  (MPPI improv +0.55)
           per-iter rew_max:  -3.37  -3.20  -3.19  -2.99  -2.95  -2.90  -2.84  -2.82
           per-iter plan_rew: -3.36  -3.42  -3.15  -2.98  -2.93  -2.86  -2.84  -2.81
           plan terms (forecast): cube_pos= -0.10  cube_quat= -0.39  finger_qpos= -0.25  contact= -1.36  wrist_delta= -0.24  qvel= -0.33  cube_angvel= -0.15
  t= 133/240  realized: pos= 17.3mm  rot=  2.8°  frac_term=0.00  rate=0.30 t/s  ETA=351s
           plan: rew=  -2.99  best_sample=  -2.90  (MPPI improv +0.49)
           per-iter rew_max:  -3.39  -3.26  -3.13  -3.01  -2.98  -2.93  -2.92  -2.90
           per-iter plan_rew: -3.36  -3.18  -3.13  -2.99  -2.96  -2.94  -2.94  -2.99
           plan terms (forecast): cube_pos= -0.10  cube_quat= -0.51  finger_qpos= -0.26  contact= -1.33  wrist_delta= -0.27  qvel= -0.34  cube_angvel= -0.19
  t= 134/240  realized: pos= 17.2mm  rot=  1.4°  frac_term=0.00  rate=0.30 t/s  ETA=348s
           plan: rew=  -2.87  best_sample=  -2.82  (MPPI improv +0.58)
           per-iter rew_max:  -3.39  -3.31  -3.14  -3.04  -2.95  -2.94  -2.88  -2.82
           per-iter plan_rew: -3.40  -3.29  -3.15  -3.04  -3.03  -3.03  -2.89  -2.87
           plan terms (forecast): cube_pos= -0.10  cube_quat= -0.45  finger_qpos= -0.26  contact= -1.34  wrist_delta= -0.25  qvel= -0.31  cube_angvel= -0.17
  t= 135/240  realized: pos= 15.7mm  rot=  3.3°  frac_term=0.00  rate=0.30 t/s  ETA=345s
           plan: rew=  -2.90  best_sample=  -2.88  (MPPI improv +0.57)
           per-iter rew_max:  -3.46  -3.25  -3.19  -3.05  -3.03  -2.98  -2.91  -2.88
           per-iter plan_rew: -3.47  -3.23  -3.35  -3.27  -3.20  -2.97  -2.91  -2.90
           plan terms (forecast): cube_pos= -0.12  cube_quat= -0.44  finger_qpos= -0.27  contact= -1.38  wrist_delta= -0.26  qvel= -0.28  cube_angvel= -0.16
  t= 136/240  realized: pos=  9.7mm  rot=  2.4°  frac_term=0.00  rate=0.30 t/s  ETA=342s
           plan: rew=  -2.93  best_sample=  -2.93  (MPPI improv +0.51)
           per-iter rew_max:  -3.44  -3.27  -3.11  -3.06  -3.01  -2.97  -2.95  -2.93
           per-iter plan_rew: -3.43  -3.21  -3.09  -3.02  -3.01  -2.97  -2.95  -2.93
           plan terms (forecast): cube_pos= -0.11  cube_quat= -0.41  finger_qpos= -0.30  contact= -1.40  wrist_delta= -0.26  qvel= -0.29  cube_angvel= -0.16
  t= 137/240  realized: pos=  9.6mm  rot=  2.3°  frac_term=0.00  rate=0.30 t/s  ETA=339s
           plan: rew=  -3.12  best_sample=  -3.12  (MPPI improv +0.56)
           per-iter rew_max:  -3.68  -3.52  -3.40  -3.29  -3.24  -3.17  -3.14  -3.12
           per-iter plan_rew: -3.66  -3.52  -3.35  -3.27  -3.21  -3.21  -3.14  -3.12
           plan terms (forecast): cube_pos= -0.10  cube_quat= -0.56  finger_qpos= -0.29  contact= -1.39  wrist_delta= -0.26  qvel= -0.31  cube_angvel= -0.21
  t= 138/240  realized: pos=  8.9mm  rot=  1.2°  frac_term=0.00  rate=0.30 t/s  ETA=336s
           plan: rew=  -3.04  best_sample=  -3.01  (MPPI improv +0.92)
           per-iter rew_max:  -3.93  -3.63  -3.44  -3.30  -3.17  -3.12  -3.04  -3.01
           per-iter plan_rew: -3.77  -3.63  -3.41  -3.25  -3.19  -3.12  -3.05  -3.04
           plan terms (forecast): cube_pos= -0.12  cube_quat= -0.49  finger_qpos= -0.30  contact= -1.44  wrist_delta= -0.23  qvel= -0.30  cube_angvel= -0.15
  t= 139/240  realized: pos=  9.6mm  rot=  1.3°  frac_term=0.00  rate=0.30 t/s  ETA=333s
           plan: rew=  -3.25  best_sample=  -3.25  (MPPI improv +0.83)
           per-iter rew_max:  -4.07  -3.91  -3.70  -3.53  -3.44  -3.32  -3.31  -3.25
           per-iter plan_rew: -4.04  -3.94  -3.65  -3.48  -3.45  -3.38  -3.37  -3.25
           plan terms (forecast): cube_pos= -0.13  cube_quat= -0.58  finger_qpos= -0.31  contact= -1.48  wrist_delta= -0.21  qvel= -0.34  cube_angvel= -0.21
  t= 140/240  realized: pos= 14.6mm  rot=  1.8°  frac_term=0.00  rate=0.30 t/s  ETA=330s
           plan: rew=  -3.40  best_sample=  -3.41  (MPPI improv +1.31)
           per-iter rew_max:  -4.72  -3.93  -3.83  -3.59  -3.55  -3.49  -3.43  -3.41
           per-iter plan_rew: -4.91  -3.92  -3.89  -3.74  -3.78  -3.48  -3.49  -3.40
           plan terms (forecast): cube_pos= -0.18  cube_quat= -0.58  finger_qpos= -0.32  contact= -1.54  wrist_delta= -0.26  qvel= -0.41  cube_angvel= -0.12
  t= 141/240  realized: pos= 18.6mm  rot=  3.2°  frac_term=0.00  rate=0.30 t/s  ETA=327s
           plan: rew=  -3.72  best_sample=  -3.72  (MPPI improv +0.38)
           per-iter rew_max:  -4.10  -4.08  -3.97  -3.98  -3.89  -3.84  -3.74  -3.72
           per-iter plan_rew: -4.10  -4.02  -3.98  -3.91  -3.93  -3.91  -3.75  -3.72
           plan terms (forecast): cube_pos= -0.22  cube_quat= -0.61  finger_qpos= -0.30  contact= -1.70  wrist_delta= -0.21  qvel= -0.41  cube_angvel= -0.27
  t= 142/240  realized: pos= 18.1mm  rot=  2.6°  frac_term=0.00  rate=0.30 t/s  ETA=323s
           plan: rew=  -4.09  best_sample=  -4.07  (MPPI improv +0.26)
           per-iter rew_max:  -4.33  -4.34  -4.34  -4.31  -4.23  -4.15  -4.12  -4.07
           per-iter plan_rew: -4.35  -4.34  -6.98  -4.49  -4.23  -4.12  -4.10  -4.09
           plan terms (forecast): cube_pos= -0.31  cube_quat= -0.60  finger_qpos= -0.28  contact= -2.05  wrist_delta= -0.20  qvel= -0.39  cube_angvel= -0.26
  t= 143/240  realized: pos= 12.3mm  rot=  0.3°  frac_term=0.00  rate=0.30 t/s  ETA=320s
           plan: rew=  -5.56  best_sample=  -5.16  (MPPI improv +1.36)
           per-iter rew_max:  -6.52  -6.04  -5.78  -5.32  -5.34  -5.17  -5.12  -5.16
           per-iter plan_rew: -6.50  -5.98  -5.50  -6.16  -6.06  -5.66  -6.95  -5.56
           plan terms (forecast): cube_pos= -0.39  cube_quat= -1.54  finger_qpos= -0.27  contact= -2.45  wrist_delta= -0.19  qvel= -0.41  cube_angvel= -0.33
  t= 144/240  realized: pos=  6.8mm  rot=  1.8°  frac_term=0.00  rate=0.30 t/s  ETA=317s
           plan: rew=  -7.33  best_sample=  -7.32  (MPPI improv -0.85)
           per-iter rew_max:  -6.47  -9.77  -8.71  -8.20  -7.72  -7.60  -7.31  -7.32
           per-iter plan_rew: -6.31  -9.76  -8.71  -8.28  -7.72  -7.59  -7.33  -7.33
           plan terms (forecast): cube_pos= -0.43  cube_quat= -3.22  finger_qpos= -0.27  contact= -2.55  wrist_delta= -0.18  qvel= -0.40  cube_angvel= -0.28
  t= 145/240  realized: pos= 10.3mm  rot=  4.5°  frac_term=1.00  rate=0.30 t/s  ETA=314s
           plan: rew=  -6.64  best_sample=  -6.68  (MPPI improv +2.44)
           per-iter rew_max:  -9.12  -7.99  -7.99  -7.24  -7.22  -7.21  -7.03  -6.68
           per-iter plan_rew: -9.10  -8.00  -7.99  -7.24  -7.33  -6.96  -7.03  -6.64
           plan terms (forecast): cube_pos= -0.53  cube_quat= -2.12  finger_qpos= -0.27  contact= -2.71  wrist_delta= -0.22  qvel= -0.40  cube_angvel= -0.38
  t= 146/240  realized: pos= 14.6mm  rot=  4.9°  frac_term=0.00  rate=0.30 t/s  ETA=311s
           plan: rew=  -7.85  best_sample=  -7.56  (MPPI improv +6.95)
           per-iter rew_max:  -14.51  -9.65  -8.94  -8.38  -8.06  -8.01  -7.78  -7.56
           per-iter plan_rew: -14.92  -9.34  -11.56  -9.05  -8.95  -8.11  -7.86  -7.85
           plan terms (forecast): cube_pos= -0.19  cube_quat= -4.65  finger_qpos= -0.29  contact= -1.56  wrist_delta= -0.28  qvel= -0.45  cube_angvel= -0.43
  t= 147/240  realized: pos= 16.0mm  rot=  5.3°  frac_term=0.00  rate=0.30 t/s  ETA=308s
           plan: rew=  -6.97  best_sample=  -6.75  (MPPI improv +1.36)
           per-iter rew_max:  -8.11  -7.47  -7.58  -7.21  -6.87  -6.78  -6.73  -6.75
           per-iter plan_rew: -7.94  -12.62  -8.98  -8.07  -7.11  -6.80  -8.09  -6.97
           plan terms (forecast): cube_pos= -0.20  cube_quat= -3.84  finger_qpos= -0.28  contact= -1.53  wrist_delta= -0.21  qvel= -0.44  cube_angvel= -0.47
  t= 148/240  realized: pos= 17.7mm  rot=  6.7°  frac_term=0.00  rate=0.30 t/s  ETA=305s
           plan: rew= -12.83  best_sample=  -6.19  (MPPI improv +0.75)
           per-iter rew_max:  -6.93  -6.95  -6.92  -6.77  -6.55  -6.13  -6.22  -6.19
           per-iter plan_rew: -7.25  -17.17  -15.12  -17.25  -8.32  -6.53  -6.21  -12.83
           plan terms (forecast): cube_pos= -0.61  cube_quat= -7.54  finger_qpos= -0.28  contact= -2.80  wrist_delta= -0.26  qvel= -0.48  cube_angvel= -0.86
  t= 149/240  realized: pos= 16.2mm  rot=  7.2°  frac_term=0.00  rate=0.30 t/s  ETA=303s
           plan: rew=  -6.35  best_sample=  -6.33  (MPPI improv +0.81)
           per-iter rew_max:  -7.15  -6.75  -6.99  -6.67  -6.62  -6.50  -6.41  -6.33
           per-iter plan_rew: -15.44  -16.05  -16.58  -9.17  -22.26  -14.74  -6.38  -6.35
           plan terms (forecast): cube_pos= -0.20  cube_quat= -3.12  finger_qpos= -0.28  contact= -1.54  wrist_delta= -0.25  qvel= -0.46  cube_angvel= -0.50
  t= 150/240  realized: pos= 12.9mm  rot=  7.8°  frac_term=0.00  rate=0.30 t/s  ETA=300s
           plan: rew=  -6.19  best_sample=  -6.08  (MPPI improv +0.83)
           per-iter rew_max:  -6.91  -7.17  -6.33  -6.30  -6.31  -6.20  -6.06  -6.08
           per-iter plan_rew: -11.98  -17.70  -6.31  -15.98  -15.76  -15.75  -8.76  -6.19
           plan terms (forecast): cube_pos= -0.20  cube_quat= -2.96  finger_qpos= -0.28  contact= -1.51  wrist_delta= -0.20  qvel= -0.48  cube_angvel= -0.57
  t= 151/240  realized: pos=  8.0mm  rot=  5.5°  frac_term=0.00  rate=0.30 t/s  ETA=297s
           plan: rew=  -6.43  best_sample=  -6.17  (MPPI improv +0.48)
           per-iter rew_max:  -6.65  -6.87  -6.79  -6.74  -6.58  -6.53  -6.17  -6.17
           per-iter plan_rew: -10.25  -16.62  -6.79  -7.57  -7.87  -6.31  -6.19  -6.43
           plan terms (forecast): cube_pos= -0.17  cube_quat= -3.25  finger_qpos= -0.28  contact= -1.51  wrist_delta= -0.23  qvel= -0.46  cube_angvel= -0.52
  t= 152/240  realized: pos= 12.0mm  rot=  6.2°  frac_term=0.00  rate=0.30 t/s  ETA=294s
           plan: rew= -10.11  best_sample=  -6.25  (MPPI improv +0.45)
           per-iter rew_max:  -6.70  -6.52  -6.54  -6.45  -6.47  -6.42  -6.30  -6.25
           per-iter plan_rew: -6.64  -6.55  -6.45  -19.82  -19.66  -8.06  -6.47  -10.11
           plan terms (forecast): cube_pos= -0.22  cube_quat= -6.67  finger_qpos= -0.27  contact= -1.51  wrist_delta= -0.21  qvel= -0.45  cube_angvel= -0.77
  t= 153/240  realized: pos= 14.8mm  rot=  8.0°  frac_term=0.00  rate=0.30 t/s  ETA=290s
           plan: rew=  -8.93  best_sample=  -6.36  (MPPI improv +0.39)
           per-iter rew_max:  -6.75  -6.55  -6.67  -6.58  -7.30  -6.51  -6.42  -6.36
           per-iter plan_rew: -6.76  -6.70  -18.65  -18.40  -18.37  -20.34  -6.44  -8.93
           plan terms (forecast): cube_pos= -0.24  cube_quat= -5.66  finger_qpos= -0.25  contact= -1.49  wrist_delta= -0.21  qvel= -0.41  cube_angvel= -0.67
  t= 154/240  realized: pos= 18.3mm  rot= 10.9°  frac_term=0.00  rate=0.30 t/s  ETA=287s
           plan: rew=  -5.64  best_sample=  -5.60  (MPPI improv +0.92)
           per-iter rew_max:  -6.53  -6.55  -6.55  -6.35  -6.15  -5.90  -5.82  -5.60
           per-iter plan_rew: -6.56  -20.51  -6.56  -6.65  -6.96  -5.92  -5.70  -5.64
           plan terms (forecast): cube_pos= -0.24  cube_quat= -2.48  finger_qpos= -0.26  contact= -1.52  wrist_delta= -0.28  qvel= -0.40  cube_angvel= -0.47
  t= 155/240  realized: pos= 19.4mm  rot= 16.2°  frac_term=0.00  rate=0.30 t/s  ETA=284s
           plan: rew=  -5.48  best_sample=  -5.41  (MPPI improv +0.94)
           per-iter rew_max:  -6.35  -6.06  -5.82  -5.75  -5.55  -5.52  -5.49  -5.41
           per-iter plan_rew: -6.32  -25.94  -25.21  -26.30  -5.54  -5.55  -28.96  -5.48
           plan terms (forecast): cube_pos= -0.23  cube_quat= -2.38  finger_qpos= -0.25  contact= -1.44  wrist_delta= -0.29  qvel= -0.39  cube_angvel= -0.50
  t= 156/240  realized: pos= 16.3mm  rot= 17.1°  frac_term=0.00  rate=0.30 t/s  ETA=281s
           plan: rew=  -4.91  best_sample=  -4.91  (MPPI improv +0.64)
           per-iter rew_max:  -5.55  -5.19  -5.35  -5.15  -5.14  -5.09  -4.93  -4.91
           per-iter plan_rew: -5.45  -21.52  -32.73  -5.15  -19.95  -5.11  -5.00  -4.91
           plan terms (forecast): cube_pos= -0.26  cube_quat= -1.77  finger_qpos= -0.27  contact= -1.45  wrist_delta= -0.33  qvel= -0.38  cube_angvel= -0.45
  t= 157/240  realized: pos= 19.0mm  rot= 19.8°  frac_term=0.00  rate=0.30 t/s  ETA=277s
           plan: rew=  -4.81  best_sample=  -4.65  (MPPI improv +0.80)
           per-iter rew_max:  -5.45  -5.42  -5.42  -5.27  -4.94  -4.83  -4.73  -4.65
           per-iter plan_rew: -5.43  -5.43  -29.80  -5.34  -4.91  -4.85  -22.67  -4.81
           plan terms (forecast): cube_pos= -0.26  cube_quat= -1.66  finger_qpos= -0.26  contact= -1.40  wrist_delta= -0.32  qvel= -0.43  cube_angvel= -0.49
  t= 158/240  realized: pos= 37.6mm  rot=  9.4°  frac_term=0.00  rate=0.30 t/s  ETA=274s
           plan: rew=  -5.83  best_sample=  -4.64  (MPPI improv +1.23)
           per-iter rew_max:  -5.87  -5.69  -5.37  -5.16  -4.92  -4.84  -4.79  -4.64
           per-iter plan_rew: -6.03  -5.67  -5.36  -5.16  -4.95  -6.35  -4.76  -5.83
           plan terms (forecast): cube_pos= -0.29  cube_quat= -2.79  finger_qpos= -0.25  contact= -1.19  wrist_delta= -0.34  qvel= -0.40  cube_angvel= -0.57
  t= 159/240  realized: pos= 39.7mm  rot=  8.2°  frac_term=0.00  rate=0.30 t/s  ETA=271s
           plan: rew=  -5.77  best_sample=  -4.66  (MPPI improv +2.64)
           per-iter rew_max:  -7.31  -5.84  -5.44  -5.09  -4.93  -4.81  -4.75  -4.66
           per-iter plan_rew: -8.75  -5.98  -5.54  -5.98  -4.93  -4.87  -4.75  -5.77
           plan terms (forecast): cube_pos= -0.27  cube_quat= -2.80  finger_qpos= -0.25  contact= -1.33  wrist_delta= -0.26  qvel= -0.39  cube_angvel= -0.47
  t= 160/240  realized: pos= 37.6mm  rot= 10.4°  frac_term=0.00  rate=0.30 t/s  ETA=268s
           plan: rew=  -3.80  best_sample=  -3.61  (MPPI improv +1.22)
           per-iter rew_max:  -4.84  -4.45  -4.20  -4.03  -3.93  -3.78  -3.65  -3.61
           per-iter plan_rew: -5.60  -4.46  -4.20  -4.08  -3.94  -3.81  -3.68  -3.80
           plan terms (forecast): cube_pos= -0.25  cube_quat= -1.17  finger_qpos= -0.26  contact= -1.23  wrist_delta= -0.22  qvel= -0.38  cube_angvel= -0.30
  t= 161/240  realized: pos= 33.9mm  rot=  6.8°  frac_term=0.00  rate=0.30 t/s  ETA=264s
           plan: rew=  -3.55  best_sample=  -3.20  (MPPI improv +0.81)
           per-iter rew_max:  -4.01  -3.80  -3.57  -3.54  -3.35  -3.28  -3.30  -3.20
           per-iter plan_rew: -4.13  -3.81  -3.57  -26.42  -5.95  -3.37  -3.41  -3.55
           plan terms (forecast): cube_pos= -0.24  cube_quat= -0.91  finger_qpos= -0.27  contact= -1.28  wrist_delta= -0.21  qvel= -0.37  cube_angvel= -0.26
  t= 162/240  realized: pos= 32.5mm  rot=  4.6°  frac_term=0.00  rate=0.30 t/s  ETA=261s
           plan: rew=  -3.72  best_sample=  -3.03  (MPPI improv +0.76)
           per-iter rew_max:  -3.79  -3.57  -3.36  -3.26  -3.19  -3.09  -3.08  -3.03
           per-iter plan_rew: -3.88  -4.24  -4.45  -3.34  -3.32  -3.33  -3.07  -3.72
           plan terms (forecast): cube_pos= -0.22  cube_quat= -1.13  finger_qpos= -0.27  contact= -1.22  wrist_delta= -0.21  qvel= -0.41  cube_angvel= -0.27
  t= 163/240  realized: pos= 26.2mm  rot=  1.9°  frac_term=0.00  rate=0.30 t/s  ETA=258s
           plan: rew=  -3.08  best_sample=  -2.81  (MPPI improv +0.67)
           per-iter rew_max:  -3.48  -3.35  -3.12  -3.04  -2.96  -2.93  -2.84  -2.81
           per-iter plan_rew: -4.01  -3.61  -3.20  -3.26  -3.14  -2.97  -2.87  -3.08
           plan terms (forecast): cube_pos= -0.24  cube_quat= -0.55  finger_qpos= -0.25  contact= -1.23  wrist_delta= -0.23  qvel= -0.40  cube_angvel= -0.19
  t= 164/240  realized: pos= 22.8mm  rot=  3.4°  frac_term=0.00  rate=0.30 t/s  ETA=254s
           plan: rew=  -2.80  best_sample=  -2.74  (MPPI improv +0.28)
           per-iter rew_max:  -3.02  -3.05  -3.08  -3.01  -2.91  -2.85  -2.82  -2.74
           per-iter plan_rew: -3.05  -3.17  -3.17  -3.00  -3.08  -2.88  -2.82  -2.80
           plan terms (forecast): cube_pos= -0.22  cube_quat= -0.41  finger_qpos= -0.24  contact= -1.19  wrist_delta= -0.19  qvel= -0.40  cube_angvel= -0.15
  t= 165/240  realized: pos= 20.9mm  rot=  1.5°  frac_term=0.00  rate=0.30 t/s  ETA=251s
           plan: rew=  -2.71  best_sample=  -2.68  (MPPI improv +0.43)
           per-iter rew_max:  -3.11  -3.00  -3.02  -2.96  -2.77  -2.77  -2.70  -2.68
           per-iter plan_rew: -3.00  -3.02  -3.04  -2.92  -2.80  -2.73  -2.72  -2.71
           plan terms (forecast): cube_pos= -0.21  cube_quat= -0.43  finger_qpos= -0.23  contact= -1.13  wrist_delta= -0.16  qvel= -0.39  cube_angvel= -0.17
  t= 166/240  realized: pos= 19.9mm  rot=  1.8°  frac_term=0.00  rate=0.30 t/s  ETA=247s
           plan: rew=  -2.62  best_sample=  -2.60  (MPPI improv +0.54)
           per-iter rew_max:  -3.13  -3.13  -3.02  -2.86  -2.78  -2.65  -2.65  -2.60
           per-iter plan_rew: -3.10  -3.03  -3.02  -2.85  -2.88  -2.67  -2.72  -2.62
           plan terms (forecast): cube_pos= -0.20  cube_quat= -0.40  finger_qpos= -0.22  contact= -1.08  wrist_delta= -0.20  qvel= -0.36  cube_angvel= -0.16
  t= 167/240  realized: pos= 23.3mm  rot=  2.9°  frac_term=0.00  rate=0.30 t/s  ETA=244s
           plan: rew=  -2.70  best_sample=  -2.65  (MPPI improv +0.73)
           per-iter rew_max:  -3.38  -3.17  -3.10  -2.89  -2.82  -2.74  -2.70  -2.65
           per-iter plan_rew: -3.39  -3.13  -3.11  -3.02  -2.85  -2.79  -2.72  -2.70
           plan terms (forecast): cube_pos= -0.21  cube_quat= -0.47  finger_qpos= -0.22  contact= -1.04  wrist_delta= -0.19  qvel= -0.38  cube_angvel= -0.18
  t= 168/240  realized: pos= 28.4mm  rot=  1.5°  frac_term=0.00  rate=0.30 t/s  ETA=240s
           plan: rew=  -2.63  best_sample=  -2.64  (MPPI improv +0.43)
           per-iter rew_max:  -3.06  -3.13  -3.04  -2.92  -2.81  -2.76  -2.64  -2.64
           per-iter plan_rew: -3.06  -3.04  -3.26  -2.96  -2.77  -2.75  -2.74  -2.63
           plan terms (forecast): cube_pos= -0.21  cube_quat= -0.38  finger_qpos= -0.22  contact= -1.03  wrist_delta= -0.21  qvel= -0.40  cube_angvel= -0.18
  t= 169/240  realized: pos= 34.1mm  rot=  0.7°  frac_term=0.00  rate=0.30 t/s  ETA=237s
           plan: rew=  -2.68  best_sample=  -2.68  (MPPI improv +0.75)
           per-iter rew_max:  -3.43  -2.97  -2.95  -2.93  -2.91  -2.89  -2.79  -2.68
           per-iter plan_rew: -3.52  -3.53  -2.93  -3.02  -3.18  -2.93  -2.89  -2.68
           plan terms (forecast): cube_pos= -0.21  cube_quat= -0.49  finger_qpos= -0.21  contact= -0.98  wrist_delta= -0.20  qvel= -0.38  cube_angvel= -0.22
  t= 170/240  realized: pos= 37.8mm  rot=  0.6°  frac_term=0.00  rate=0.30 t/s  ETA=233s
           plan: rew=  -2.81  best_sample=  -2.80  (MPPI improv +0.84)
           per-iter rew_max:  -3.64  -3.35  -3.22  -3.09  -3.01  -2.88  -2.86  -2.80
           per-iter plan_rew: -3.58  -3.30  -3.18  -3.07  -2.92  -2.94  -2.84  -2.81
           plan terms (forecast): cube_pos= -0.22  cube_quat= -0.46  finger_qpos= -0.23  contact= -1.11  wrist_delta= -0.21  qvel= -0.37  cube_angvel= -0.21
  t= 171/240  realized: pos= 37.4mm  rot=  1.2°  frac_term=0.00  rate=0.30 t/s  ETA=230s
           plan: rew=  -2.71  best_sample=  -2.72  (MPPI improv +0.91)
           per-iter rew_max:  -3.62  -3.30  -3.08  -2.88  -2.83  -2.82  -2.79  -2.72
           per-iter plan_rew: -3.56  -3.34  -2.86  -2.85  -3.11  -2.79  -2.77  -2.71
           plan terms (forecast): cube_pos= -0.22  cube_quat= -0.40  finger_qpos= -0.24  contact= -1.13  wrist_delta= -0.16  qvel= -0.38  cube_angvel= -0.18
  t= 172/240  realized: pos= 35.3mm  rot=  1.4°  frac_term=0.00  rate=0.30 t/s  ETA=227s
           plan: rew=  -2.72  best_sample=  -2.60  (MPPI improv +0.22)
           per-iter rew_max:  -2.81  -2.80  -2.79  -2.77  -2.74  -2.68  -2.65  -2.60
           per-iter plan_rew: -2.80  -2.80  -2.77  -2.83  -2.72  -2.71  -2.72  -2.72
           plan terms (forecast): cube_pos= -0.20  cube_quat= -0.49  finger_qpos= -0.22  contact= -1.16  wrist_delta= -0.15  qvel= -0.35  cube_angvel= -0.15
  t= 173/240  realized: pos= 31.5mm  rot=  1.9°  frac_term=0.00  rate=0.30 t/s  ETA=223s
           plan: rew=  -2.53  best_sample=  -2.52  (MPPI improv +0.61)
           per-iter rew_max:  -3.13  -3.01  -2.99  -2.74  -2.68  -2.60  -2.54  -2.52
           per-iter plan_rew: -5.88  -3.08  -2.85  -2.69  -2.84  -2.65  -2.60  -2.53
           plan terms (forecast): cube_pos= -0.22  cube_quat= -0.28  finger_qpos= -0.24  contact= -1.15  wrist_delta= -0.16  qvel= -0.35  cube_angvel= -0.13
  t= 174/240  realized: pos= 26.8mm  rot=  1.7°  frac_term=0.00  rate=0.30 t/s  ETA=220s
           plan: rew=  -3.08  best_sample=  -2.73  (MPPI improv +0.59)
           per-iter rew_max:  -3.32  -2.87  -2.87  -2.81  -2.81  -2.81  -2.74  -2.73
           per-iter plan_rew: -2.87  -2.87  -2.82  -2.83  -2.81  -3.58  -2.74  -3.08
           plan terms (forecast): cube_pos= -0.21  cube_quat= -0.69  finger_qpos= -0.23  contact= -1.28  wrist_delta= -0.16  qvel= -0.36  cube_angvel= -0.15
  t= 175/240  realized: pos= 22.3mm  rot=  1.9°  frac_term=0.00  rate=0.30 t/s  ETA=216s
           plan: rew=  -2.64  best_sample=  -2.66  (MPPI improv +0.17)
           per-iter rew_max:  -2.83  -2.78  -2.78  -2.84  -2.72  -2.71  -2.66  -2.66
           per-iter plan_rew: -2.81  -2.78  -2.88  -2.83  -2.72  -2.69  -2.70  -2.64
           plan terms (forecast): cube_pos= -0.19  cube_quat= -0.33  finger_qpos= -0.22  contact= -1.28  wrist_delta= -0.15  qvel= -0.33  cube_angvel= -0.15
  t= 176/240  realized: pos= 21.7mm  rot=  1.3°  frac_term=0.00  rate=0.30 t/s  ETA=213s
           plan: rew=  -2.63  best_sample=  -2.64  (MPPI improv +0.29)
           per-iter rew_max:  -2.94  -2.96  -2.91  -2.86  -2.78  -2.70  -2.67  -2.64
           per-iter plan_rew: -2.96  -2.95  -2.94  -2.83  -2.73  -2.67  -2.68  -2.63
           plan terms (forecast): cube_pos= -0.18  cube_quat= -0.34  finger_qpos= -0.22  contact= -1.26  wrist_delta= -0.16  qvel= -0.32  cube_angvel= -0.16
  t= 177/240  realized: pos= 19.6mm  rot=  1.7°  frac_term=0.00  rate=0.30 t/s  ETA=209s
           plan: rew=  -2.63  best_sample=  -2.63  (MPPI improv +0.24)
           per-iter rew_max:  -2.86  -2.85  -2.85  -2.78  -2.73  -2.67  -2.65  -2.63
           per-iter plan_rew: -2.86  -2.85  -2.79  -2.78  -2.72  -2.75  -2.64  -2.63
           plan terms (forecast): cube_pos= -0.19  cube_quat= -0.31  finger_qpos= -0.21  contact= -1.31  wrist_delta= -0.19  qvel= -0.29  cube_angvel= -0.12
  t= 178/240  realized: pos= 16.4mm  rot=  2.2°  frac_term=0.00  rate=0.30 t/s  ETA=206s
           plan: rew=  -2.76  best_sample=  -2.75  (MPPI improv +0.18)
           per-iter rew_max:  -2.93  -2.94  -2.85  -2.85  -2.85  -2.82  -2.78  -2.75
           per-iter plan_rew: -2.94  -2.85  -2.85  -2.85  -2.82  -2.83  -2.78  -2.76
           plan terms (forecast): cube_pos= -0.19  cube_quat= -0.38  finger_qpos= -0.20  contact= -1.36  wrist_delta= -0.16  qvel= -0.28  cube_angvel= -0.17
  t= 179/240  realized: pos= 15.0mm  rot=  3.1°  frac_term=0.00  rate=0.30 t/s  ETA=203s
           plan: rew=  -2.77  best_sample=  -2.76  (MPPI improv +0.39)
           per-iter rew_max:  -3.15  -3.13  -3.10  -3.07  -2.91  -2.86  -2.79  -2.76
           per-iter plan_rew: -3.13  -3.10  -3.09  -3.00  -2.92  -2.83  -2.79  -2.77
           plan terms (forecast): cube_pos= -0.20  cube_quat= -0.40  finger_qpos= -0.20  contact= -1.41  wrist_delta= -0.16  qvel= -0.25  cube_angvel= -0.16
  t= 180/240  realized: pos= 17.2mm  rot=  2.3°  frac_term=0.00  rate=0.30 t/s  ETA=199s
           plan: rew=  -2.74  best_sample=  -2.74  (MPPI improv +0.34)
           per-iter rew_max:  -3.08  -3.07  -3.04  -3.01  -2.89  -2.82  -2.78  -2.74
           per-iter plan_rew: -3.07  -3.07  -3.03  -2.99  -2.88  -2.86  -2.76  -2.74
           plan terms (forecast): cube_pos= -0.20  cube_quat= -0.30  finger_qpos= -0.20  contact= -1.49  wrist_delta= -0.18  qvel= -0.23  cube_angvel= -0.14
  t= 181/240  realized: pos= 19.8mm  rot=  2.0°  frac_term=0.00  rate=0.30 t/s  ETA=196s
           plan: rew=  -2.87  best_sample=  -2.86  (MPPI improv +0.43)
           per-iter rew_max:  -3.29  -3.28  -3.19  -3.03  -3.00  -2.93  -2.87  -2.86
           per-iter plan_rew: -3.28  -3.19  -3.18  -3.03  -3.00  -2.90  -2.90  -2.87
           plan terms (forecast): cube_pos= -0.21  cube_quat= -0.29  finger_qpos= -0.20  contact= -1.65  wrist_delta= -0.16  qvel= -0.22  cube_angvel= -0.14
  t= 182/240  realized: pos= 24.2mm  rot=  2.5°  frac_term=0.00  rate=0.30 t/s  ETA=192s
           plan: rew=  -2.88  best_sample=  -2.89  (MPPI improv +0.47)
           per-iter rew_max:  -3.36  -3.34  -3.18  -3.07  -3.02  -2.96  -2.91  -2.89
           per-iter plan_rew: -3.42  -3.27  -3.12  -3.08  -2.99  -2.94  -2.90  -2.88
           plan terms (forecast): cube_pos= -0.20  cube_quat= -0.26  finger_qpos= -0.21  contact= -1.72  wrist_delta= -0.15  qvel= -0.22  cube_angvel= -0.11
  t= 183/240  realized: pos= 27.4mm  rot=  0.7°  frac_term=0.00  rate=0.30 t/s  ETA=189s
           plan: rew=  -2.99  best_sample=  -3.00  (MPPI improv +0.49)
           per-iter rew_max:  -3.49  -3.31  -3.27  -3.17  -3.11  -3.06  -3.03  -3.00
           per-iter plan_rew: -3.31  -3.34  -3.27  -3.15  -3.07  -3.08  -3.03  -2.99
           plan terms (forecast): cube_pos= -0.18  cube_quat= -0.29  finger_qpos= -0.21  contact= -1.82  wrist_delta= -0.15  qvel= -0.21  cube_angvel= -0.13
  t= 184/240  realized: pos= 29.9mm  rot=  0.7°  frac_term=0.00  rate=0.30 t/s  ETA=186s
           plan: rew=  -2.94  best_sample=  -2.94  (MPPI improv +0.45)
           per-iter rew_max:  -3.39  -3.34  -3.23  -3.05  -3.02  -3.02  -2.97  -2.94
           per-iter plan_rew: -3.39  -3.25  -3.21  -3.05  -3.03  -3.02  -2.95  -2.94
           plan terms (forecast): cube_pos= -0.19  cube_quat= -0.25  finger_qpos= -0.21  contact= -1.83  wrist_delta= -0.15  qvel= -0.20  cube_angvel= -0.11
  t= 185/240  realized: pos= 27.7mm  rot=  0.3°  frac_term=0.00  rate=0.30 t/s  ETA=182s
           plan: rew=  -3.10  best_sample=  -3.10  (MPPI improv +0.32)
           per-iter rew_max:  -3.42  -3.42  -3.41  -3.29  -3.24  -3.16  -3.12  -3.10
           per-iter plan_rew: -3.42  -3.41  -3.38  -3.29  -3.20  -3.16  -3.10  -3.10
           plan terms (forecast): cube_pos= -0.20  cube_quat= -0.31  finger_qpos= -0.21  contact= -1.91  wrist_delta= -0.14  qvel= -0.19  cube_angvel= -0.14
  t= 186/240  realized: pos= 19.0mm  rot=  0.7°  frac_term=0.00  rate=0.30 t/s  ETA=179s
           plan: rew=  -3.23  best_sample=  -3.24  (MPPI improv +0.47)
           per-iter rew_max:  -3.71  -3.59  -3.51  -3.43  -3.38  -3.30  -3.26  -3.24
           per-iter plan_rew: -3.69  -3.55  -3.48  -3.39  -3.35  -3.30  -3.26  -3.23
           plan terms (forecast): cube_pos= -0.19  cube_quat= -0.29  finger_qpos= -0.21  contact= -2.05  wrist_delta= -0.17  qvel= -0.18  cube_angvel= -0.13
  t= 187/240  realized: pos= 15.1mm  rot=  0.6°  frac_term=0.00  rate=0.30 t/s  ETA=176s
           plan: rew=  -3.16  best_sample=  -3.17  (MPPI improv +0.53)
           per-iter rew_max:  -3.70  -3.70  -3.55  -3.38  -3.31  -3.28  -3.23  -3.17
           per-iter plan_rew: -3.70  -3.68  -3.55  -3.38  -3.31  -3.36  -3.21  -3.16
           plan terms (forecast): cube_pos= -0.20  cube_quat= -0.34  finger_qpos= -0.22  contact= -1.93  wrist_delta= -0.14  qvel= -0.19  cube_angvel= -0.14
  t= 188/240  realized: pos= 13.5mm  rot=  1.3°  frac_term=0.00  rate=0.30 t/s  ETA=172s
           plan: rew=  -3.08  best_sample=  -3.07  (MPPI improv +0.51)
           per-iter rew_max:  -3.58  -3.53  -3.41  -3.30  -3.20  -3.15  -3.13  -3.07
           per-iter plan_rew: -3.58  -3.45  -3.33  -3.28  -3.22  -3.14  -3.09  -3.08
           plan terms (forecast): cube_pos= -0.19  cube_quat= -0.38  finger_qpos= -0.21  contact= -1.79  wrist_delta= -0.17  qvel= -0.23  cube_angvel= -0.11
  t= 189/240  realized: pos=  9.9mm  rot=  1.3°  frac_term=0.00  rate=0.30 t/s  ETA=169s
           plan: rew=  -2.93  best_sample=  -2.93  (MPPI improv +0.70)
           per-iter rew_max:  -3.63  -3.36  -3.16  -3.13  -3.05  -3.02  -2.97  -2.93
           per-iter plan_rew: -3.58  -3.33  -3.16  -3.07  -3.02  -2.99  -2.97  -2.93
           plan terms (forecast): cube_pos= -0.21  cube_quat= -0.24  finger_qpos= -0.24  contact= -1.70  wrist_delta= -0.18  qvel= -0.26  cube_angvel= -0.11
  t= 190/240  realized: pos= 10.0mm  rot=  0.6°  frac_term=0.00  rate=0.30 t/s  ETA=165s
           plan: rew=  -2.90  best_sample=  -2.87  (MPPI improv +0.83)
           per-iter rew_max:  -3.70  -3.54  -3.29  -3.19  -3.05  -2.97  -2.92  -2.87
           per-iter plan_rew: -3.64  -3.43  -3.25  -3.13  -3.03  -2.94  -2.90  -2.90
           plan terms (forecast): cube_pos= -0.20  cube_quat= -0.30  finger_qpos= -0.24  contact= -1.55  wrist_delta= -0.23  qvel= -0.25  cube_angvel= -0.11
  t= 191/240  realized: pos= 12.7mm  rot=  1.3°  frac_term=0.00  rate=0.30 t/s  ETA=162s
           plan: rew=  -2.93  best_sample=  -2.91  (MPPI improv +0.57)
           per-iter rew_max:  -3.48  -3.15  -3.14  -3.07  -3.02  -2.97  -2.93  -2.91
           per-iter plan_rew: -3.63  -3.15  -3.11  -3.05  -3.02  -2.99  -2.93  -2.93
           plan terms (forecast): cube_pos= -0.19  cube_quat= -0.31  finger_qpos= -0.23  contact= -1.51  wrist_delta= -0.27  qvel= -0.28  cube_angvel= -0.14
  t= 192/240  realized: pos= 14.5mm  rot=  0.5°  frac_term=0.00  rate=0.30 t/s  ETA=158s
           plan: rew=  -2.99  best_sample=  -2.97  (MPPI improv +0.39)
           per-iter rew_max:  -3.37  -3.37  -3.31  -3.20  -3.12  -3.05  -3.00  -2.97
           per-iter plan_rew: -3.37  -3.36  -3.26  -3.17  -3.14  -3.05  -3.01  -2.99
           plan terms (forecast): cube_pos= -0.19  cube_quat= -0.42  finger_qpos= -0.23  contact= -1.45  wrist_delta= -0.24  qvel= -0.28  cube_angvel= -0.18
  t= 193/240  realized: pos= 18.1mm  rot=  1.1°  frac_term=0.00  rate=0.30 t/s  ETA=155s
           plan: rew=  -2.96  best_sample=  -2.95  (MPPI improv +0.74)
           per-iter rew_max:  -3.69  -3.54  -3.22  -3.17  -3.15  -3.03  -3.00  -2.95
           per-iter plan_rew: -3.62  -4.93  -3.22  -3.26  -3.12  -3.17  -3.07  -2.96
           plan terms (forecast): cube_pos= -0.19  cube_quat= -0.49  finger_qpos= -0.22  contact= -1.41  wrist_delta= -0.19  qvel= -0.27  cube_angvel= -0.18
  t= 194/240  realized: pos= 22.5mm  rot=  1.5°  frac_term=0.00  rate=0.30 t/s  ETA=152s
           plan: rew=  -2.70  best_sample=  -2.72  (MPPI improv +0.68)
           per-iter rew_max:  -3.40  -3.22  -3.09  -2.95  -2.87  -2.80  -2.75  -2.72
           per-iter plan_rew: -3.26  -3.10  -3.00  -2.92  -2.84  -2.78  -2.74  -2.70
           plan terms (forecast): cube_pos= -0.18  cube_quat= -0.33  finger_qpos= -0.23  contact= -1.36  wrist_delta= -0.19  qvel= -0.29  cube_angvel= -0.11
  t= 195/240  realized: pos= 25.3mm  rot=  1.8°  frac_term=0.00  rate=0.30 t/s  ETA=148s
           plan: rew=  -2.64  best_sample=  -2.61  (MPPI improv +0.48)
           per-iter rew_max:  -3.09  -2.98  -2.91  -2.74  -2.69  -2.65  -2.64  -2.61
           per-iter plan_rew: -2.98  -2.91  -2.91  -2.74  -2.70  -2.71  -2.70  -2.64
           plan terms (forecast): cube_pos= -0.18  cube_quat= -0.26  finger_qpos= -0.24  contact= -1.31  wrist_delta= -0.22  qvel= -0.31  cube_angvel= -0.11
  t= 196/240  realized: pos= 22.3mm  rot=  1.1°  frac_term=0.00  rate=0.30 t/s  ETA=145s
           plan: rew=  -2.53  best_sample=  -2.53  (MPPI improv +0.67)
           per-iter rew_max:  -3.20  -3.04  -2.87  -2.71  -2.64  -2.59  -2.55  -2.53
           per-iter plan_rew: -3.12  -2.94  -2.82  -2.81  -2.67  -2.60  -2.54  -2.53
           plan terms (forecast): cube_pos= -0.18  cube_quat= -0.22  finger_qpos= -0.23  contact= -1.29  wrist_delta= -0.19  qvel= -0.32  cube_angvel= -0.10
  t= 197/240  realized: pos= 20.1mm  rot=  1.1°  frac_term=0.00  rate=0.30 t/s  ETA=142s
           plan: rew=  -2.45  best_sample=  -2.45  (MPPI improv +0.45)
           per-iter rew_max:  -2.90  -2.81  -2.75  -2.60  -2.52  -2.47  -2.46  -2.45
           per-iter plan_rew: -4.20  -2.76  -2.74  -2.61  -2.50  -2.47  -2.46  -2.45
           plan terms (forecast): cube_pos= -0.19  cube_quat= -0.25  finger_qpos= -0.23  contact= -1.19  wrist_delta= -0.19  qvel= -0.31  cube_angvel= -0.09
  t= 198/240  realized: pos= 20.4mm  rot=  1.9°  frac_term=0.00  rate=0.30 t/s  ETA=138s
           plan: rew=  -2.39  best_sample=  -2.33  (MPPI improv +0.34)
           per-iter rew_max:  -2.67  -2.59  -2.63  -2.46  -2.42  -2.37  -2.36  -2.33
           per-iter plan_rew: -2.60  -2.63  -2.51  -2.45  -2.43  -2.42  -2.33  -2.39
           plan terms (forecast): cube_pos= -0.19  cube_quat= -0.29  finger_qpos= -0.21  contact= -1.09  wrist_delta= -0.17  qvel= -0.33  cube_angvel= -0.10
  t= 199/240  realized: pos= 20.3mm  rot=  1.4°  frac_term=0.00  rate=0.30 t/s  ETA=135s
           plan: rew=  -2.30  best_sample=  -2.29  (MPPI improv +0.33)
           per-iter rew_max:  -2.62  -2.62  -2.52  -2.53  -2.40  -2.33  -2.29  -2.29
           per-iter plan_rew: -2.63  -2.64  -5.64  -2.74  -2.50  -2.32  -2.37  -2.30
           plan terms (forecast): cube_pos= -0.20  cube_quat= -0.27  finger_qpos= -0.21  contact= -0.99  wrist_delta= -0.16  qvel= -0.35  cube_angvel= -0.12
  t= 200/240  realized: pos= 20.8mm  rot=  1.5°  frac_term=0.00  rate=0.30 t/s  ETA=131s
           plan: rew=  -2.36  best_sample=  -2.30  (MPPI improv +0.03)
           per-iter rew_max:  -2.34  -2.34  -2.34  -2.34  -2.34  -2.34  -2.30  -2.30
           per-iter plan_rew: -2.34  -2.34  -2.34  -2.34  -2.34  -2.35  -2.31  -2.36
           plan terms (forecast): cube_pos= -0.20  cube_quat= -0.30  finger_qpos= -0.20  contact= -1.01  wrist_delta= -0.15  qvel= -0.37  cube_angvel= -0.14
  t= 201/240  realized: pos= 23.8mm  rot=  1.5°  frac_term=0.00  rate=0.30 t/s  ETA=128s
           plan: rew=  -2.24  best_sample=  -2.24  (MPPI improv +0.15)
           per-iter rew_max:  -2.38  -2.39  -2.53  -2.46  -2.31  -2.30  -2.26  -2.24
           per-iter plan_rew: -3.47  -3.48  -2.49  -2.40  -2.47  -2.26  -2.25  -2.24
           plan terms (forecast): cube_pos= -0.19  cube_quat= -0.26  finger_qpos= -0.20  contact= -0.96  wrist_delta= -0.17  qvel= -0.34  cube_angvel= -0.12
  t= 202/240  realized: pos= 28.0mm  rot=  0.7°  frac_term=0.00  rate=0.30 t/s  ETA=125s
           plan: rew=  -2.12  best_sample=  -2.13  (MPPI improv +0.23)
           per-iter rew_max:  -2.36  -2.36  -2.31  -2.18  -2.18  -2.18  -2.17  -2.13
           per-iter plan_rew: -2.36  -2.33  -2.20  -2.18  -2.18  -2.17  -2.15  -2.12
           plan terms (forecast): cube_pos= -0.19  cube_quat= -0.25  finger_qpos= -0.19  contact= -0.89  wrist_delta= -0.15  qvel= -0.33  cube_angvel= -0.11
  t= 203/240  realized: pos= 25.3mm  rot=  1.7°  frac_term=0.00  rate=0.30 t/s  ETA=121s
           plan: rew=  -1.99  best_sample=  -2.01  (MPPI improv +0.18)
           per-iter rew_max:  -2.19  -2.15  -2.15  -2.15  -2.12  -2.07  -2.03  -2.01
           per-iter plan_rew: -2.27  -2.15  -2.15  -2.12  -2.12  -2.05  -2.06  -1.99
           plan terms (forecast): cube_pos= -0.19  cube_quat= -0.26  finger_qpos= -0.18  contact= -0.83  wrist_delta= -0.12  qvel= -0.32  cube_angvel= -0.10
  t= 204/240  realized: pos= 21.8mm  rot=  2.3°  frac_term=0.00  rate=0.30 t/s  ETA=118s
           plan: rew=  -1.94  best_sample=  -1.93  (MPPI improv +0.15)
           per-iter rew_max:  -2.08  -2.08  -2.08  -2.12  -2.02  -1.97  -1.95  -1.93
           per-iter plan_rew: -2.09  -2.08  -2.12  -2.04  -2.15  -1.96  -1.96  -1.94
           plan terms (forecast): cube_pos= -0.18  cube_quat= -0.23  finger_qpos= -0.18  contact= -0.84  wrist_delta= -0.12  qvel= -0.29  cube_angvel= -0.08
  t= 205/240  realized: pos= 16.9mm  rot=  2.3°  frac_term=0.00  rate=0.30 t/s  ETA=115s
           plan: rew=  -1.90  best_sample=  -1.92  (MPPI improv +0.15)
           per-iter rew_max:  -2.06  -2.06  -2.02  -2.04  -2.01  -1.97  -1.93  -1.92
           per-iter plan_rew: -2.06  -2.05  -2.05  -2.01  -2.01  -2.00  -1.92  -1.90
           plan terms (forecast): cube_pos= -0.17  cube_quat= -0.23  finger_qpos= -0.17  contact= -0.84  wrist_delta= -0.12  qvel= -0.28  cube_angvel= -0.10
  t= 206/240  realized: pos= 13.4mm  rot=  1.8°  frac_term=0.00  rate=0.30 t/s  ETA=111s
           plan: rew=  -1.87  best_sample=  -1.87  (MPPI improv +0.14)
           per-iter rew_max:  -2.01  -2.01  -1.98  -1.99  -1.99  -1.91  -1.89  -1.87
           per-iter plan_rew: -2.01  -2.09  -2.09  -2.02  -1.97  -1.89  -1.87  -1.87
           plan terms (forecast): cube_pos= -0.15  cube_quat= -0.22  finger_qpos= -0.17  contact= -0.83  wrist_delta= -0.14  qvel= -0.27  cube_angvel= -0.09
  t= 207/240  realized: pos= 14.0mm  rot=  1.0°  frac_term=0.00  rate=0.30 t/s  ETA=108s
           plan: rew=  -1.80  best_sample=  -1.83  (MPPI improv +0.34)
           per-iter rew_max:  -2.17  -1.94  -1.92  -1.96  -1.87  -1.87  -1.86  -1.83
           per-iter plan_rew: -1.93  -1.93  -1.96  -1.98  -1.93  -1.89  -1.88  -1.80
           plan terms (forecast): cube_pos= -0.15  cube_quat= -0.20  finger_qpos= -0.16  contact= -0.80  wrist_delta= -0.13  qvel= -0.27  cube_angvel= -0.08
  t= 208/240  realized: pos= 15.4mm  rot=  0.7°  frac_term=0.00  rate=0.30 t/s  ETA=105s
           plan: rew=  -1.82  best_sample=  -1.82  (MPPI improv +0.20)
           per-iter rew_max:  -2.02  -2.04  -2.02  -1.97  -1.89  -1.87  -1.85  -1.82
           per-iter plan_rew: -2.05  -2.02  -1.96  -2.00  -1.88  -1.85  -1.85  -1.82
           plan terms (forecast): cube_pos= -0.14  cube_quat= -0.20  finger_qpos= -0.16  contact= -0.82  wrist_delta= -0.13  qvel= -0.29  cube_angvel= -0.08
  t= 209/240  realized: pos= 17.1mm  rot=  1.5°  frac_term=0.00  rate=0.30 t/s  ETA=101s
           plan: rew=  -1.77  best_sample=  -1.78  (MPPI improv +0.19)
           per-iter rew_max:  -1.97  -2.04  -1.97  -1.82  -1.83  -1.83  -1.79  -1.78
           per-iter plan_rew: -2.10  -2.04  -1.85  -1.83  -1.85  -1.82  -1.78  -1.77
           plan terms (forecast): cube_pos= -0.12  cube_quat= -0.18  finger_qpos= -0.15  contact= -0.79  wrist_delta= -0.13  qvel= -0.29  cube_angvel= -0.09
  t= 210/240  realized: pos= 16.9mm  rot=  2.0°  frac_term=0.00  rate=0.30 t/s  ETA=98s
           plan: rew=  -1.78  best_sample=  -1.77  (MPPI improv +0.20)
           per-iter rew_max:  -1.97  -1.97  -2.02  -1.97  -1.91  -1.83  -1.79  -1.77
           per-iter plan_rew: -2.11  -2.05  -2.09  -2.16  -1.95  -1.79  -1.81  -1.78
           plan terms (forecast): cube_pos= -0.12  cube_quat= -0.22  finger_qpos= -0.14  contact= -0.79  wrist_delta= -0.14  qvel= -0.26  cube_angvel= -0.10
  t= 211/240  realized: pos= 17.7mm  rot=  0.5°  frac_term=0.00  rate=0.30 t/s  ETA=95s
           plan: rew=  -1.66  best_sample=  -1.64  (MPPI improv +0.45)
           per-iter rew_max:  -2.10  -1.99  -1.86  -1.85  -1.80  -1.71  -1.67  -1.64
           per-iter plan_rew: -2.01  -1.87  -1.85  -1.84  -1.78  -1.72  -1.68  -1.66
           plan terms (forecast): cube_pos= -0.12  cube_quat= -0.21  finger_qpos= -0.14  contact= -0.72  wrist_delta= -0.13  qvel= -0.24  cube_angvel= -0.10
  t= 212/240  realized: pos= 20.7mm  rot=  1.2°  frac_term=0.00  rate=0.30 t/s  ETA=91s
           plan: rew=  -1.73  best_sample=  -1.76  (MPPI improv +0.50)
           per-iter rew_max:  -2.26  -2.17  -2.06  -2.00  -1.92  -1.82  -1.79  -1.76
           per-iter plan_rew: -2.22  -2.07  -2.06  -1.99  -1.90  -1.85  -1.77  -1.73
           plan terms (forecast): cube_pos= -0.13  cube_quat= -0.23  finger_qpos= -0.16  contact= -0.74  wrist_delta= -0.13  qvel= -0.25  cube_angvel= -0.10
  t= 213/240  realized: pos= 22.3mm  rot=  1.1°  frac_term=0.00  rate=0.30 t/s  ETA=88s
           plan: rew=  -1.64  best_sample=  -1.64  (MPPI improv +0.77)
           per-iter rew_max:  -2.40  -2.00  -1.96  -1.88  -1.77  -1.72  -1.67  -1.64
           per-iter plan_rew: -2.15  -2.05  -1.95  -1.84  -1.80  -1.77  -1.65  -1.64
           plan terms (forecast): cube_pos= -0.13  cube_quat= -0.18  finger_qpos= -0.15  contact= -0.73  wrist_delta= -0.16  qvel= -0.23  cube_angvel= -0.07
  t= 214/240  realized: pos= 20.8mm  rot=  1.4°  frac_term=0.00  rate=0.30 t/s  ETA=84s
           plan: rew=  -1.60  best_sample=  -1.59  (MPPI improv +0.27)
           per-iter rew_max:  -1.86  -1.88  -1.86  -1.85  -1.74  -1.67  -1.63  -1.59
           per-iter plan_rew: -1.88  -1.86  -1.87  -1.77  -1.70  -1.72  -1.64  -1.60
           plan terms (forecast): cube_pos= -0.11  cube_quat= -0.18  finger_qpos= -0.15  contact= -0.73  wrist_delta= -0.14  qvel= -0.22  cube_angvel= -0.08
  t= 215/240  realized: pos= 20.7mm  rot=  0.4°  frac_term=0.00  rate=0.30 t/s  ETA=81s
           plan: rew=  -1.57  best_sample=  -1.58  (MPPI improv +0.19)
           per-iter rew_max:  -1.77  -1.77  -1.77  -1.74  -1.72  -1.67  -1.63  -1.58
           per-iter plan_rew: -1.77  -1.77  -1.75  -1.75  -1.68  -1.67  -1.64  -1.57
           plan terms (forecast): cube_pos= -0.10  cube_quat= -0.19  finger_qpos= -0.14  contact= -0.73  wrist_delta= -0.13  qvel= -0.19  cube_angvel= -0.09
  t= 216/240  realized: pos= 20.0mm  rot=  0.4°  frac_term=0.00  rate=0.30 t/s  ETA=78s
           plan: rew=  -1.52  best_sample=  -1.52  (MPPI improv +0.23)
           per-iter rew_max:  -1.76  -1.76  -1.70  -1.67  -1.64  -1.58  -1.54  -1.52
           per-iter plan_rew: -1.76  -1.75  -1.67  -1.66  -1.60  -1.55  -1.53  -1.52
           plan terms (forecast): cube_pos= -0.10  cube_quat= -0.20  finger_qpos= -0.14  contact= -0.70  wrist_delta= -0.12  qvel= -0.17  cube_angvel= -0.07
  t= 217/240  realized: pos= 17.7mm  rot=  0.9°  frac_term=0.00  rate=0.30 t/s  ETA=74s
           plan: rew=  -1.50  best_sample=  -1.48  (MPPI improv +0.24)
           per-iter rew_max:  -1.72  -1.72  -1.72  -1.66  -1.62  -1.55  -1.50  -1.48
           per-iter plan_rew: -1.75  -1.72  -1.74  -1.67  -1.57  -1.51  -1.52  -1.50
           plan terms (forecast): cube_pos= -0.09  cube_quat= -0.24  finger_qpos= -0.14  contact= -0.68  wrist_delta= -0.11  qvel= -0.17  cube_angvel= -0.07
  t= 218/240  realized: pos= 15.6mm  rot=  0.5°  frac_term=0.00  rate=0.30 t/s  ETA=71s
           plan: rew=  -1.42  best_sample=  -1.42  (MPPI improv +0.25)
           per-iter rew_max:  -1.67  -1.67  -1.66  -1.63  -1.54  -1.48  -1.44  -1.42
           per-iter plan_rew: -1.69  -1.67  -1.63  -1.64  -1.52  -1.46  -1.46  -1.42
           plan terms (forecast): cube_pos= -0.08  cube_quat= -0.21  finger_qpos= -0.14  contact= -0.66  wrist_delta= -0.10  qvel= -0.15  cube_angvel= -0.07
  t= 219/240  realized: pos= 14.0mm  rot=  0.4°  frac_term=0.00  rate=0.30 t/s  ETA=68s
           plan: rew=  -1.32  best_sample=  -1.32  (MPPI improv +0.30)
           per-iter rew_max:  -1.62  -1.62  -1.54  -1.48  -1.43  -1.36  -1.34  -1.32
           per-iter plan_rew: -1.65  -1.61  -1.49  -1.49  -1.42  -1.36  -1.34  -1.32
           plan terms (forecast): cube_pos= -0.07  cube_quat= -0.14  finger_qpos= -0.13  contact= -0.63  wrist_delta= -0.14  qvel= -0.15  cube_angvel= -0.06
  t= 220/240  realized: pos= 13.3mm  rot=  0.6°  frac_term=0.00  rate=0.30 t/s  ETA=64s
           plan: rew=  -1.32  best_sample=  -1.32  (MPPI improv +0.22)
           per-iter rew_max:  -1.54  -1.52  -1.50  -1.43  -1.42  -1.36  -1.34  -1.32
           per-iter plan_rew: -1.53  -1.55  -1.43  -1.42  -1.41  -1.37  -1.36  -1.32
           plan terms (forecast): cube_pos= -0.07  cube_quat= -0.19  finger_qpos= -0.13  contact= -0.60  wrist_delta= -0.11  qvel= -0.14  cube_angvel= -0.08
  t= 221/240  realized: pos= 13.7mm  rot=  0.4°  frac_term=0.00  rate=0.30 t/s  ETA=61s
           plan: rew=  -1.25  best_sample=  -1.25  (MPPI improv +0.27)
           per-iter rew_max:  -1.52  -1.52  -1.49  -1.38  -1.36  -1.31  -1.28  -1.25
           per-iter plan_rew: -1.56  -1.50  -1.43  -1.37  -1.33  -1.29  -1.28  -1.25
           plan terms (forecast): cube_pos= -0.07  cube_quat= -0.17  finger_qpos= -0.13  contact= -0.57  wrist_delta= -0.11  qvel= -0.14  cube_angvel= -0.06
  t= 222/240  realized: pos= 12.2mm  rot=  0.8°  frac_term=0.00  rate=0.30 t/s  ETA=58s
           plan: rew=  -1.23  best_sample=  -1.21  (MPPI improv +0.23)
           per-iter rew_max:  -1.45  -1.45  -1.44  -1.36  -1.33  -1.29  -1.24  -1.21
           per-iter plan_rew: -1.45  -1.44  -1.37  -1.34  -1.30  -1.25  -1.24  -1.23
           plan terms (forecast): cube_pos= -0.07  cube_quat= -0.19  finger_qpos= -0.13  contact= -0.55  wrist_delta= -0.09  qvel= -0.14  cube_angvel= -0.06
  t= 223/240  realized: pos=  8.4mm  rot=  0.6°  frac_term=0.00  rate=0.30 t/s  ETA=54s
           plan: rew=  -1.12  best_sample=  -1.13  (MPPI improv +0.26)
           per-iter rew_max:  -1.39  -1.39  -1.36  -1.27  -1.20  -1.17  -1.14  -1.13
           per-iter plan_rew: -1.41  -1.37  -1.28  -1.20  -1.20  -1.17  -1.15  -1.12
           plan terms (forecast): cube_pos= -0.06  cube_quat= -0.13  finger_qpos= -0.13  contact= -0.52  wrist_delta= -0.11  qvel= -0.12  cube_angvel= -0.05
  t= 224/240  realized: pos=  6.6mm  rot=  0.8°  frac_term=0.00  rate=0.30 t/s  ETA=51s
           plan: rew=  -1.11  best_sample=  -1.11  (MPPI improv +0.22)
           per-iter rew_max:  -1.32  -1.32  -1.33  -1.28  -1.18  -1.15  -1.14  -1.11
           per-iter plan_rew: -1.33  -1.33  -1.31  -1.21  -1.17  -1.15  -1.15  -1.11
           plan terms (forecast): cube_pos= -0.05  cube_quat= -0.15  finger_qpos= -0.13  contact= -0.51  wrist_delta= -0.11  qvel= -0.12  cube_angvel= -0.05
  t= 225/240  realized: pos=  6.3mm  rot=  0.5°  frac_term=0.00  rate=0.30 t/s  ETA=47s
           plan: rew=  -1.01  best_sample=  -1.02  (MPPI improv +0.17)
           per-iter rew_max:  -1.19  -1.20  -1.19  -1.18  -1.11  -1.07  -1.04  -1.02
           per-iter plan_rew: -1.20  -1.20  -1.19  -1.11  -1.10  -1.05  -1.05  -1.01
           plan terms (forecast): cube_pos= -0.05  cube_quat= -0.14  finger_qpos= -0.12  contact= -0.41  wrist_delta= -0.11  qvel= -0.11  cube_angvel= -0.06
  t= 226/240  realized: pos=  6.1mm  rot=  0.5°  frac_term=0.00  rate=0.30 t/s  ETA=44s
           plan: rew=  -0.98  best_sample=  -0.97  (MPPI improv +0.20)
           per-iter rew_max:  -1.17  -1.16  -1.12  -1.10  -1.00  -0.99  -0.99  -0.97
           per-iter plan_rew: -1.17  -1.16  -1.11  -1.04  -0.99  -0.99  -0.97  -0.98
           plan terms (forecast): cube_pos= -0.05  cube_quat= -0.14  finger_qpos= -0.12  contact= -0.36  wrist_delta= -0.15  qvel= -0.11  cube_angvel= -0.05
  t= 227/240  realized: pos=  5.9mm  rot=  0.3°  frac_term=0.00  rate=0.30 t/s  ETA=41s
           plan: rew=  -1.00  best_sample=  -1.01  (MPPI improv +0.16)
           per-iter rew_max:  -1.17  -1.16  -1.16  -1.16  -1.15  -1.07  -1.04  -1.01
           per-iter plan_rew: -1.16  -1.16  -1.16  -1.16  -1.11  -1.06  -1.03  -1.00
           plan terms (forecast): cube_pos= -0.05  cube_quat= -0.18  finger_qpos= -0.12  contact= -0.33  wrist_delta= -0.13  qvel= -0.10  cube_angvel= -0.08
  t= 228/240  realized: pos=  5.7mm  rot=  0.4°  frac_term=0.00  rate=0.30 t/s  ETA=37s
           plan: rew=  -0.89  best_sample=  -0.89  (MPPI improv +0.28)
           per-iter rew_max:  -1.17  -1.17  -1.12  -1.05  -1.01  -0.97  -0.93  -0.89
           per-iter plan_rew: -1.17  -1.17  -1.11  -1.01  -1.00  -0.95  -0.92  -0.89
           plan terms (forecast): cube_pos= -0.04  cube_quat= -0.12  finger_qpos= -0.13  contact= -0.31  wrist_delta= -0.16  qvel= -0.10  cube_angvel= -0.03
  t= 229/240  realized: pos=  5.4mm  rot=  0.4°  frac_term=0.00  rate=0.30 t/s  ETA=34s
           plan: rew=  -0.89  best_sample=  -0.89  (MPPI improv +0.26)
           per-iter rew_max:  -1.16  -1.15  -1.14  -1.03  -0.96  -0.93  -0.92  -0.89
           per-iter plan_rew: -1.15  -1.14  -1.03  -1.00  -0.95  -0.93  -0.90  -0.89
           plan terms (forecast): cube_pos= -0.04  cube_quat= -0.12  finger_qpos= -0.13  contact= -0.27  wrist_delta= -0.18  qvel= -0.10  cube_angvel= -0.05
  t= 230/240  realized: pos=  5.4mm  rot=  0.6°  frac_term=0.00  rate=0.30 t/s  ETA=30s
           plan: rew=  -0.91  best_sample=  -0.91  (MPPI improv +0.17)
           per-iter rew_max:  -1.08  -1.07  -1.07  -1.01  -0.99  -0.95  -0.93  -0.91
           per-iter plan_rew: -1.09  -1.07  -1.07  -1.01  -0.95  -0.94  -0.92  -0.91
           plan terms (forecast): cube_pos= -0.04  cube_quat= -0.15  finger_qpos= -0.13  contact= -0.24  wrist_delta= -0.18  qvel= -0.10  cube_angvel= -0.06
  t= 231/240  realized: pos=  5.8mm  rot=  0.6°  frac_term=0.00  rate=0.30 t/s  ETA=27s
           plan: rew=  -0.85  best_sample=  -0.85  (MPPI improv +0.21)
           per-iter rew_max:  -1.06  -1.09  -1.07  -1.00  -0.95  -0.90  -0.88  -0.85
           per-iter plan_rew: -1.09  -1.08  -1.00  -0.97  -0.92  -0.89  -0.86  -0.85
           plan terms (forecast): cube_pos= -0.05  cube_quat= -0.13  finger_qpos= -0.13  contact= -0.22  wrist_delta= -0.17  qvel= -0.10  cube_angvel= -0.05
  t= 232/240  realized: pos=  4.8mm  rot=  0.5°  frac_term=0.00  rate=0.30 t/s  ETA=24s
           plan: rew=  -0.79  best_sample=  -0.79  (MPPI improv +0.23)
           per-iter rew_max:  -1.02  -0.99  -0.94  -0.91  -0.88  -0.86  -0.82  -0.79
           per-iter plan_rew: -0.99  -0.95  -0.91  -0.91  -0.88  -0.84  -0.81  -0.79
           plan terms (forecast): cube_pos= -0.04  cube_quat= -0.13  finger_qpos= -0.13  contact= -0.19  wrist_delta= -0.16  qvel= -0.09  cube_angvel= -0.04
  t= 233/240  realized: pos=  4.0mm  rot=  0.4°  frac_term=0.00  rate=0.30 t/s  ETA=20s
           plan: rew=  -0.76  best_sample=  -0.77  (MPPI improv +0.28)
           per-iter rew_max:  -1.05  -1.05  -0.97  -0.89  -0.84  -0.81  -0.79  -0.77
           per-iter plan_rew: -1.05  -0.97  -0.91  -0.85  -0.81  -0.80  -0.78  -0.76
           plan terms (forecast): cube_pos= -0.05  cube_quat= -0.12  finger_qpos= -0.13  contact= -0.17  wrist_delta= -0.16  qvel= -0.10  cube_angvel= -0.04
  t= 234/240  realized: pos=  3.6mm  rot=  0.3°  frac_term=0.00  rate=0.30 t/s  ETA=17s
           plan: rew=  -0.74  best_sample=  -0.74  (MPPI improv +0.20)
           per-iter rew_max:  -0.94  -0.92  -0.90  -0.87  -0.81  -0.77  -0.76  -0.74
           per-iter plan_rew: -0.93  -0.90  -0.87  -0.82  -0.78  -0.76  -0.74  -0.74
           plan terms (forecast): cube_pos= -0.05  cube_quat= -0.12  finger_qpos= -0.13  contact= -0.15  wrist_delta= -0.14  qvel= -0.10  cube_angvel= -0.05
  t= 235/240  realized: pos=  3.3mm  rot=  0.5°  frac_term=0.00  rate=0.30 t/s  ETA=14s
           plan: rew=  -0.70  best_sample=  -0.70  (MPPI improv +0.18)
           per-iter rew_max:  -0.87  -0.86  -0.84  -0.81  -0.79  -0.75  -0.74  -0.70
           per-iter plan_rew: -0.86  -0.84  -0.83  -0.79  -0.77  -0.74  -0.73  -0.70
           plan terms (forecast): cube_pos= -0.05  cube_quat= -0.12  finger_qpos= -0.12  contact= -0.13  wrist_delta= -0.13  qvel= -0.10  cube_angvel= -0.05
  t= 236/240  realized: pos=  2.7mm  rot=  0.3°  frac_term=0.00  rate=0.30 t/s  ETA=10s
           plan: rew=  -0.68  best_sample=  -0.69  (MPPI improv +0.20)
           per-iter rew_max:  -0.88  -0.88  -0.86  -0.83  -0.79  -0.75  -0.73  -0.69
           per-iter plan_rew: -0.88  -0.86  -0.83  -0.82  -0.78  -0.73  -0.71  -0.68
           plan terms (forecast): cube_pos= -0.06  cube_quat= -0.13  finger_qpos= -0.12  contact= -0.10  wrist_delta= -0.13  qvel= -0.09  cube_angvel= -0.05
  t= 237/240  realized: pos=  2.9mm  rot=  0.5°  frac_term=0.00  rate=0.30 t/s  ETA=7s
           plan: rew=  -0.65  best_sample=  -0.65  (MPPI improv +0.19)
           per-iter rew_max:  -0.84  -0.84  -0.83  -0.77  -0.75  -0.70  -0.68  -0.65
           per-iter plan_rew: -0.84  -0.83  -0.83  -0.75  -0.70  -0.69  -0.67  -0.65
           plan terms (forecast): cube_pos= -0.06  cube_quat= -0.14  finger_qpos= -0.12  contact= -0.08  wrist_delta= -0.10  qvel= -0.10  cube_angvel= -0.05
  t= 238/240  realized: pos=  3.7mm  rot=  0.3°  frac_term=0.00  rate=0.30 t/s  ETA=3s
           plan: rew=  -0.60  best_sample=  -0.61  (MPPI improv +0.39)
           per-iter rew_max:  -1.00  -0.73  -0.73  -0.72  -0.70  -0.66  -0.64  -0.61
           per-iter plan_rew: -0.73  -0.73  -0.72  -0.72  -0.66  -0.65  -0.62  -0.60
           plan terms (forecast): cube_pos= -0.06  cube_quat= -0.12  finger_qpos= -0.12  contact= -0.05  wrist_delta= -0.09  qvel= -0.10  cube_angvel= -0.04
  t= 239/240  realized: pos=  5.2mm  rot=  0.2°  frac_term=0.00  rate=0.30 t/s  ETA=0s
           plan: rew=  -0.59  best_sample=  -0.60  (MPPI improv +0.16)
           per-iter rew_max:  -0.76  -0.75  -0.75  -0.74  -0.70  -0.67  -0.62  -0.60
           per-iter plan_rew: -0.76  -0.75  -0.75  -0.72  -0.69  -0.65  -0.62  -0.59
           plan terms (forecast): cube_pos= -0.07  cube_quat= -0.14  finger_qpos= -0.12  contact= -0.03  wrist_delta= -0.08  qvel= -0.10  cube_angvel= -0.05
  Done in 810.9s (0.30 ticks/s).
  ── error summary ──
  cube pos err (mm):  mean= 18.9  max= 45.2
  cube rot err (°):   mean=  2.6  max= 19.8
  iter improvement:   mean=+0.54  max=+6.95
  mean per-term reward across ticks (more-negative = bigger penalty):
    cube_pos      -0.187
    cube_quat     -0.956
    finger_qpos   -0.199
    contact       -1.217
    wrist_delta   -0.184
    qvel          -0.283
    cube_angvel   -0.209
