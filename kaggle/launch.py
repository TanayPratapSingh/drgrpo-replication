"""Package the code as a private Kaggle dataset and push the kernel.

    python kaggle/launch.py smoke            # 3 real steps: measure speed, memory, vLLM
    python kaggle/launch.py full --seeds 0   # both arms, 100 steps each
    python kaggle/launch.py status | logs | output

Uses the kaggle CLI, which reads the user's own ~/.kaggle credentials; this
script never reads or prints them.
"""
import argparse, json, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "kaggle" / "build"
KAGGLE = str(ROOT / ".kvenv" / "bin" / "kaggle")
DATASET_SLUG = "drgrpo-code"
KERNEL_SLUG = "drgrpo-replication"


def run(args, capture=False):
    print("$ kaggle " + " ".join(args), flush=True)
    r = subprocess.run([KAGGLE, *args], capture_output=capture, text=True)
    if capture:
        return r
    if r.returncode:
        sys.exit(r.returncode)


def username():
    r = subprocess.run([KAGGLE, "config", "view"], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.strip().startswith("- username:"):
            return line.split(":", 1)[1].strip()
    sys.exit("could not determine the Kaggle username; is the CLI configured?")


def push_code(user):
    d = BUILD / "dataset"
    shutil.rmtree(d, ignore_errors=True)
    (d / "src").mkdir(parents=True)
    for f in ("fetch_data.py", "template.py", "variants.py", "objective_torch.py", "train_torch.py", "__init__.py"):
        shutil.copy(ROOT / "src" / f, d / "src" / f)
    shutil.copytree(ROOT / "src" / "vendor", d / "src" / "vendor",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (d / "dataset-metadata.json").write_text(json.dumps({
        "title": "drgrpo code", "id": f"{user}/{DATASET_SLUG}", "licenses": [{"name": "MIT"}]}))
    exists = run(["datasets", "status", f"{user}/{DATASET_SLUG}"], capture=True).returncode == 0
    if exists:
        run(["datasets", "version", "-p", str(d), "-m", "update code", "-r", "zip", "-q"])
    else:
        run(["datasets", "create", "-p", str(d), "-r", "zip", "-q"])


def push_kernel(user, config, accelerator, resume):
    d = BUILD / "kernel"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    src = (ROOT / "kaggle" / "run_kernel.py").read_text().replace("CONFIG = {}", f"CONFIG = json.loads({json.dumps(json.dumps(config))})", 1)
    (d / "run_kernel.py").write_text(src)
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{KERNEL_SLUG}", "title": KERNEL_SLUG, "code_file": "run_kernel.py",
        "language": "python", "kernel_type": "script", "is_private": "true",
        "enable_gpu": "true", "enable_internet": "true", "machine_shape": accelerator,
        "dataset_sources": [f"{user}/{DATASET_SLUG}"],
        "kernel_sources": [f"{user}/{KERNEL_SLUG}"] if resume else [],
        "competition_sources": [], "model_sources": []}, indent=2))
    run(["kernels", "push", "-p", str(d), "--accelerator", accelerator])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke", "full", "status", "logs", "output"])
    ap.add_argument("--accelerator", default="NvidiaTeslaT4")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--arms", default="grpo,drgrpo")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--resume", action="store_true", help="attach the previous session's output")
    ap.add_argument("--extra", default="", help="extra args passed to train_torch.py")
    a = ap.parse_args()
    user = username()
    ref = f"{user}/{KERNEL_SLUG}"
    if a.mode == "status":
        return run(["kernels", "status", ref])
    if a.mode == "logs":
        return run(["kernels", "logs", ref])
    if a.mode == "output":
        out = ROOT / "kaggle" / "output"
        out.mkdir(parents=True, exist_ok=True)
        return run(["kernels", "output", ref, "-p", str(out), "-o"])
    if a.mode == "smoke":
        config = {"arms": ["--loss grpo --seed 0 --steps 3 --eval-n 16 --final-eval-n 16 --out smoke_grpo"],
                  "session_hours": 2.0, "try_vllm": True, "extra": a.extra}
    else:
        arms = [f"--loss {loss} --seed {s} --steps {a.steps}" for s in a.seeds.split(",") for loss in a.arms.split(",")]
        config = {"arms": arms, "session_hours": 11.3, "try_vllm": True, "extra": a.extra}
    push_code(user)
    push_kernel(user, config, a.accelerator, a.resume)


if __name__ == "__main__":
    main()
