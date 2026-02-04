# Run baseline (equal weights)
python train_multi_objective.py --experiment baseline --timesteps 100000

# another terminal, launch TensorBoard
tensorboard --logdir models/multi_objective/baseline_equal_weights/tensorboard

# Run signal-focused (aggressive collection)
python train_multi_objective.py --experiment signal_focused --timesteps 100000

# Run survival-focused (conservative play)
python train_multi_objective.py --experiment survival_focused --timesteps 100000

# Run all experiments
python train_multi_objective.py --experiment all --timesteps 100000