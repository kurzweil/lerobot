python lerobot/scripts/visualize_dataset_html.py \\n  --repo-id kkurzweil/so100_test

python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --robot.cameras='{}' \
  --control.type=teleoperate

python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=replay \
  --control.fps=30 \
  --control.repo_id=kkurzweil/so100_test \
  --control.episode=0

python lerobot/scripts/train.py \
  --dataset.repo_id=kkurzweil/so100_wrist_01 \
  --policy.type=diffusion \
  --output_dir=outputs/train/diffusion_so100_test \
  --job_name=diffusion_so100_test \
  --policy.device=mps \
  --wandb.enable=false

python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=record \
  --control.single_task="Put the Lego brick in the bin." \
  --control.fps=30 \
  --control.repo_id=kkurzweil/so100_wrist_03 \
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=60 \
  --control.reset_time_s=60 \
  --control.num_episodes=150 \
  --control.push_to_hub=true
  
python lerobot/scripts/train.py \
  --dataset.repo_id=kkurzweil/so100_wrist_02 \
  --policy.type=diffusion \
  --output_dir=outputs/train/diffusion_so100_test_02 \
  --job_name=diffusion_so100_test_02 \
  --policy.device=cuda \
  --wandb.enable=true
