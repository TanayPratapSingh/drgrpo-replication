"""Kaggle kernel entry point for the Dr. GRPO replication (PyTorch backend).

kaggle/launch.py writes the CONFIG block below before each push. The kernel
installs dependencies, copies the code from the attached dataset, fetches the
authors' data, then runs src/train_torch.py for each arm in turn. Everything
lands in /kaggle/working, which becomes the kernel's downloadable output; a
later session can attach that output and resume from its checkpoints.
"""
import glob, json, os, shutil, subprocess, sys, time

CONFIG = {}  # replaced by launch.py

T0 = time.time()
WORK = "/kaggle/working"
os.chdir(WORK)


def sh(cmd, check=True, timeout=None):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, timeout=timeout)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {cmd}")
    return r.returncode


print("CONFIG", json.dumps(CONFIG), flush=True)
sh("nvidia-smi --query-gpu=name,memory.total --format=csv", check=False)
sh(f"{sys.executable} -c \"import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'gpus',torch.cuda.device_count(),'bf16',torch.cuda.is_available() and torch.cuda.is_bf16_supported())\"", check=False)

sh(f"{sys.executable} -m pip install -q peft 'math-verify[antlr4_13_2]' latex2sympy2_extended pylatexenc")
vllm_ok = False
if CONFIG.get("try_vllm", True):
    try:
        vllm_ok = sh(f"{sys.executable} -m pip install -q vllm", check=False, timeout=1500) == 0 and \
                  sh(f"{sys.executable} -c 'import vllm; print(\"vllm\", vllm.__version__)'", check=False) == 0
    except subprocess.TimeoutExpired:
        print("vllm install timed out", flush=True)
print(f"vllm available: {vllm_ok}", flush=True)
# Kaggle ships torchao 0.10; peft refuses to import alongside any torchao older than 0.16,
# and nothing here uses it.
sh(f"{sys.executable} -m pip uninstall -y -q torchao", check=False)

sh("find /kaggle/input -maxdepth 4 | head -40", check=False)
code = os.path.join(WORK, "code")
# the dataset may hold src/ as a folder or, uploaded in zip mode, as src.zip
found = sorted(glob.glob("/kaggle/input/**/train_torch.py", recursive=True))
if found:
    shutil.copytree(os.path.dirname(found[0]), os.path.join(code, "src"), dirs_exist_ok=True)
else:
    zips = sorted(glob.glob("/kaggle/input/**/src.zip", recursive=True))
    if not zips:
        raise SystemExit("code dataset not found: no train_torch.py or src.zip under /kaggle/input")
    shutil.unpack_archive(zips[0], os.path.join(code, "src"))
    nested = glob.glob(os.path.join(code, "src", "**", "train_torch.py"), recursive=True)
    if nested and os.path.dirname(nested[0]) != os.path.join(code, "src"):
        shutil.copytree(os.path.dirname(nested[0]), os.path.join(code, "src"), dirs_exist_ok=True)
assert os.path.exists(os.path.join(code, "src", "train_torch.py")), "code layout unexpected"
print("code ready:", sorted(os.listdir(os.path.join(code, "src"))), flush=True)
sh(f"cd {code} && {sys.executable} src/fetch_data.py")

# inputs mount at /kaggle/input/<kind>/<user>/<slug>/..., so search at any depth
prev_runs = sorted(p for p in glob.glob("/kaggle/input/**/runs", recursive=True) if os.path.isdir(p))
resume_arg = f"--resume-from {prev_runs[0]}" if prev_runs else ""
if prev_runs:
    print(f"previous session output found: {prev_runs[0]}", flush=True)

budget_h = CONFIG.get("session_hours", 11.3)
for arm in CONFIG["arms"]:
    left = budget_h - (time.time() - T0) / 3600
    if left < 0.5:
        print(f"session budget spent, not starting {arm}", flush=True)
        break
    sampler = "auto" if vllm_ok else "hf"
    cmd = (f"cd {code} && {sys.executable} src/train_torch.py {arm} --sampler {sampler} "
           f"--runs-dir {WORK}/runs --data-dir {code}/data --max-hours {left:.2f} {resume_arg} "
           f"{CONFIG.get('extra', '')}")
    sh(cmd, check=False)
print(f"kernel finished in {(time.time() - T0) / 3600:.2f} h", flush=True)
