"""
LIBERO-Plus Smoke Test: OpenVLA on Camera Viewpoints L1.
Usage:
    CUDA_VISIBLE_DEVICES=2 python eval_openvla_liberoplus_smoke.py
"""
import argparse, io, json, math, os, re, sys, time
os.environ["MUJOCO_GL"] = "egl"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

sys.path.insert(0, "/root/autodl-tmp/openvla_deps")

import numpy as np
import torch
from PIL import Image

hf_dir = "/root/autodl-tmp/openvla-repo/prismatic/extern/hf"
init_path = os.path.join(hf_dir, "__init__.py")
if not os.path.exists(init_path):
    open(init_path, "w").close()
sys.path.insert(0, "/root/autodl-tmp/openvla-repo/prismatic/extern")
from hf.configuration_prismatic import OpenVLAConfig
from hf.modeling_prismatic import OpenVLAForActionPrediction
from hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

DEVICE = torch.device("cuda:0")
CHECKPOINT_PATH = "/root/autodl-tmp/openvla-libero-object"
UNNORM_KEY = "libero_object"
TASK_CLASSIFICATION = "/root/autodl-tmp/LIBERO-plus/libero/libero/benchmark/task_classification.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_object")
    p.add_argument("--category", default="Camera Viewpoints")
    p.add_argument("--difficulty", type=int, default=1)
    p.add_argument("--max_tasks", type=int, default=3)
    p.add_argument("--num_episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--num_steps_wait", type=int, default=10)
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


def load_model():
    print("[*] Loading OpenVLA from:", CHECKPOINT_PATH, flush=True)
    config = OpenVLAConfig.from_pretrained(CHECKPOINT_PATH)
    vla = OpenVLAForActionPrediction.from_pretrained(
        CHECKPOINT_PATH, config=config,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).to(DEVICE)
    with open(os.path.join(CHECKPOINT_PATH, "dataset_statistics.json")) as f:
        vla.norm_stats = json.load(f)
    vla.eval()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
    image_processor = PrismaticImageProcessor.from_pretrained(CHECKPOINT_PATH)
    processor = PrismaticProcessor(image_processor=image_processor, tokenizer=tokenizer)
    print(f"[*] Model loaded, GPU: {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)
    return vla, processor


def center_crop_image(image, crop_scale=0.9):
    w, h = image.size
    new_w = int(w * math.sqrt(crop_scale))
    new_h = int(h * math.sqrt(crop_scale))
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    cropped = image.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), Image.LANCZOS)


def preprocess_image(obs):
    img = obs["agentview_image"]
    img = img[::-1, ::-1]
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    buf.seek(0)
    pil_img = Image.open(buf).convert("RGB")
    pil_img = pil_img.resize((224, 224), Image.LANCZOS)
    pil_img = center_crop_image(pil_img, crop_scale=0.9)
    return pil_img


def get_action(vla, processor, image, task_label):
    prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
    action = vla.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False)
    return action


def normalize_gripper_action(action, binarize=True):
    action[..., -1] = 2 * action[..., -1] - 1
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    return action


def invert_gripper_action(action):
    action[..., -1] *= -1.0
    return action


def get_filtered_task_indices(suite_name, category, difficulty):
    with open(TASK_CLASSIFICATION) as f:
        tc = json.load(f)
    filtered = []
    for t in tc[suite_name]:
        if t.get("category") == category and t.get("difficulty_level") == difficulty:
            filtered.append(t["id"] - 1)
    return filtered


def get_clean_language(task_name, bddl_files_dir, problem_folder):
    """Extract clean language from BDDL file, stripping perturbation suffixes."""
    import libero.libero.envs.bddl_utils as BDDLUtils
    base_name = task_name
    if "_view_" in base_name:
        base_name = base_name.split("_view_")[0]
    if "_noise_" in base_name:
        base_name = base_name.split("_noise_")[0]
    bddl_path = os.path.join(bddl_files_dir, problem_folder, base_name + ".bddl")
    if os.path.exists(bddl_path):
        try:
            info = BDDLUtils.get_problem_info(bddl_path)
            return info["language_instruction"]
        except Exception:
            pass
    return " ".join(base_name.split("_"))


def load_init_states(task, init_states_dir):
    """Load init states with _view_ path handling and weights_only=False."""
    init_file = task.init_states_file
    if "_view_" in init_file:
        base = init_file.split("_view_")[0]
        ext = init_file.split(".")[-1]
        init_file = f"{base}.{ext}"
    elif "_table_" in init_file:
        init_file = re.sub(r'_table_\d+', '', init_file)
    elif "_tb_" in init_file:
        init_file = re.sub(r'_tb_\d+', '', init_file)
    elif "_light_" in init_file:
        base = init_file.split("_light_")[0]
        ext = init_file.split(".")[-1]
        init_file = f"{base}.{ext}"

    if "_add_" in task.init_states_file or "_level" in task.init_states_file:
        path = os.path.join(init_states_dir, "libero_newobj", task.problem_folder, task.init_states_file)
    else:
        path = os.path.join(init_states_dir, task.problem_folder, init_file)

    init_states = torch.load(path, weights_only=False)
    if "_add_" in task.init_states_file or "_level" in task.init_states_file:
        init_states = init_states.reshape(1, -1)
    return init_states


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import OffScreenRenderEnv

    init_states_dir = get_libero_path("init_states")
    bddl_files_dir = get_libero_path("bddl_files")

    task_indices = get_filtered_task_indices(args.suite, args.category, args.difficulty)
    task_indices = task_indices[:args.max_tasks]
    print(f"[*] {args.category} L{args.difficulty}: evaluating {len(task_indices)} tasks from {args.suite}", flush=True)

    task_suite = benchmark.get_benchmark_dict()[args.suite]()
    max_steps = 280

    vla, processor = load_model()

    total_successes = 0
    total_episodes = 0
    per_task_results = {}
    t_start = time.time()

    for i, task_id in enumerate(task_indices):
        task = task_suite.get_task(task_id)
        task_description = get_clean_language(task.name, bddl_files_dir, task.problem_folder)
        task_bddl_file = os.path.join(bddl_files_dir, task.problem_folder, task.bddl_file)
        print(f"\n[Task {i+1}/{len(task_indices)}] idx={task_id} | {task.name[:80]}", flush=True)
        print(f"  Language: {task_description}", flush=True)

        try:
            env = OffScreenRenderEnv(
                bddl_file_name=task_bddl_file, camera_heights=256, camera_widths=256
            )
        except Exception as e:
            print(f"  [ERROR] Failed to create env: {e}", flush=True)
            continue

        env.seed(0)
        task_successes = 0

        init_states = load_init_states(task, init_states_dir)

        for ep in range(args.num_episodes):
            env.reset()
            ep_idx = min(ep, len(init_states) - 1)
            obs = env.set_init_state(init_states[ep_idx])
            t = 0

            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step([0, 0, 0, 0, 0, 0, -1])
                        t += 1
                        continue
                    img = preprocess_image(obs)
                    action = get_action(vla, processor, img, task_description)
                    action = normalize_gripper_action(action, binarize=True)
                    action = invert_gripper_action(action)
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1
                except Exception as e:
                    print(f"  Exception t={t}: {e}", flush=True)
                    break

            total_episodes += 1
            elapsed = time.time() - t_start
            eps_per_min = total_episodes / (elapsed / 60)
            print(f"  Ep {ep+1}/{args.num_episodes} sr={task_successes}/{ep+1} "
                  f"({eps_per_min:.1f} eps/min)", flush=True)

        task_sr = task_successes / args.num_episodes
        per_task_results[task.name] = {
            "success_rate": task_sr,
            "successes": task_successes,
            "episodes": args.num_episodes,
        }
        print(f"  => Task SR: {task_sr:.3f} ({task_successes}/{args.num_episodes})", flush=True)
        env.close()

    overall_sr = total_successes / max(total_episodes, 1)
    elapsed = time.time() - t_start
    print(f"\n{'='*60}", flush=True)
    print(f"LIBERO-Plus Smoke Test: {args.category} L{args.difficulty}", flush=True)
    print(f"Overall SR: {overall_sr:.4f} ({total_successes}/{total_episodes})", flush=True)
    print(f"Time: {elapsed/60:.1f} min", flush=True)
    print(f"{'='*60}", flush=True)

    results = {
        "model": "openvla-7b-libero-object",
        "benchmark": "LIBERO-Plus",
        "category": args.category,
        "difficulty_level": args.difficulty,
        "suite": args.suite,
        "overall_success_rate": overall_sr,
        "total_successes": total_successes,
        "total_episodes": total_episodes,
        "per_task": per_task_results,
        "max_tasks": args.max_tasks,
        "num_episodes": args.num_episodes,
        "max_steps": max_steps,
        "seed": args.seed,
        "elapsed_minutes": elapsed / 60,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out_dir = "/root/autodl-tmp/importance_margin_vla_compress/eval_results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.output or f"{out_dir}/openvla_liberoplus_{args.category.replace(' ', '_').lower()}_L{args.difficulty}_smoke.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[*] Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
