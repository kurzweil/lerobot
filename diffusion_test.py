import time
import torch
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config
from lerobot.common.robot_devices.robots.configs import So100RobotConfig


so100 = So100RobotConfig()
robot = make_robot_from_config(so100)
robot.connect()

inference_time_s = 30
fps = 30
device = "mps"  # TODO: On Mac, use "mps" or "cpu"

ckpt_path = "/Users/kkurzweil/Google Drive/My Drive/lerobot/outputs/train/diffusion_so100_test_02/checkpoints/040000/pretrained_model"
#ckpt_path = "/Users/kkurzweil/Google Drive/My Drive/lerobot/outputs/train/act_so100_test_02/checkpoints/last/pretrained_model"
#ckpt_path = "/Users/kkurzweil/Google Drive/My Drive/lerobot/outputs/train/act_so100_test_02/checkpoints/180000/pretrained_model"
#ckpt_path = "/Users/kkurzweil/Google Drive/My Drive/development/lerobot/outputs/train/act_so100_test_02/checkpoints/last/pretrained_model/"
policy = DiffusionPolicy.from_pretrained(ckpt_path)
policy.to(device)

while True:
    print("loop")
    action = torch.tensor([5.8008, 128.4082, 120.2344, 72.8613, -82.0020, 14.0781])
    robot.send_action(action)
    busy_wait(2.000)
    for _ in range(inference_time_s * fps):
        start_time = time.perf_counter()

        # Read the follower state and access the frames from the cameras
        observation = robot.capture_observation()

        # Convert to pytorch format: channel first and float32 in [0,1]
        # with batch dimension
        for name in observation:
            if "image" in name:
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)
        # Remove batch dimension
        action = action.squeeze(0)
        # Move to cpu, if not already the case
        action = action.to("cpu")
        # Order the robot to move
        robot.send_action(action)

        dt_s = time.perf_counter() - start_time
        busy_wait(1 / fps - dt_s)
