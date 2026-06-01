"""Run official OFT eval code on 5 episodes."""
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "./openvla-oft-official")

from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    pretrained_checkpoint: str = "./openvla-oft-libero-object"
    task_suite_name: str = "libero_object"
    model_family: str = "openvla"
    center_crop: bool = True
    num_images_in_input: int = 2
    use_proprio: bool = True
    use_film: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    num_open_loop_steps: int = 8
    num_steps_wait: int = 10
    unnorm_key: str = "libero_object_no_noops"
    env_img_res: int = 256

cfg = Config()

from experiments.robot.openvla_utils import get_vla, get_action_head, get_vla_action
from experiments.robot.robot_utils import get_action, normalize_gripper_action, invert_gripper_action
from experiments.robot.libero.run_libero_eval import prepare_observation, process_action, TASK_MAX_STEPS
from experiments.robot.libero.libero_utils import get_libero_env, get_libero_dummy_action
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

from collections import deque
import numpy as np
import torch

print("Loading model via official code...")
model = get_vla(cfg)
processor = model.processor if hasattr(model, 'processor') else None
if processor is None:
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)

# Load action head
action_head = get_action_head(cfg, model.llm_dim)

# Load proprio projector  
from experiments.robot.openvla_utils import find_checkpoint_file, load_component_state_dict
from prismatic.models.projectors import ProprioProjector
pp_path = find_checkpoint_file(cfg.pretrained_checkpoint, "proprio_projector")
pp = ProprioProjector(llm_dim=model.llm_dim, proprio_dim=8)
pp.load_state_dict(load_component_state_dict(pp_path))
pp = pp.to(torch.device("cuda:0"), dtype=torch.bfloat16).eval()

from libero.libero import benchmark
from experiments.robot.openvla_utils import get_image_resize_size
resize_size = 224

bm = benchmark.get_benchmark_dict()["libero_object"]()
task = bm.get_task(0)
print(f"Task: {task.language}")

env, task_desc = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
init_states = bm.get_task_init_states(0)

max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

for ep in range(5):
    env.reset()
    obs = env.set_init_state(init_states[ep])
    
    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    t = 0
    success = False
    
    while t < max_steps + cfg.num_steps_wait:
        if t < cfg.num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action(cfg.model_family))
            t += 1
            continue
        
        observation, _ = prepare_observation(obs, resize_size)
        
        if len(action_queue) == 0:
            actions = get_action(
                cfg, model, observation, task_desc,
                processor=processor, action_head=action_head,
                proprio_projector=pp,
            )
            action_queue.extend(actions)
        
        action = action_queue.popleft()
        action = process_action(action, cfg.model_family)
        obs, _, done, _ = env.step(action.tolist())
        
        if done:
            success = True
            break
        t += 1
    
    print(f"  Ep {ep+1}/5 | {'OK' if success else 'FAIL'} | steps={t-cfg.num_steps_wait}")

env.close()
print("Done.")
