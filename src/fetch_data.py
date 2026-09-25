"""Fetch the paper's own train and eval splits from sail-sg/understand-r1-zero.

Using the authors' published arrow files means the question set is identical to
the one behind Figure 5, rather than a re-derivation of MATH that might differ.
"""
import pathlib
from urllib.request import urlopen

BASE = "https://raw.githubusercontent.com/sail-sg/understand-r1-zero/main/datasets"
FILES = {
    "math_12k": "train/math_12k/train/data-00000-of-00001.arrow",
    "gsm_8k": "train/gsm_8k/train/data-00000-of-00001.arrow",
    "asdiv_2k": "train/asdiv_2k/train/data-00000-of-00001.arrow",
    "eval_math500": "evaluation_suite/math/data-00000-of-00001.arrow",
}

def main():
    out = pathlib.Path(__file__).resolve().parent.parent / "data"
    out.mkdir(exist_ok=True)
    for name, rel in FILES.items():
        dest = out / f"{name}.arrow"
        if dest.exists():
            print(f"cached   {name}"); continue
        with urlopen(f"{BASE}/{rel}", timeout=120) as r:
            dest.write_bytes(r.read())
        print(f"fetched  {name}  {dest.stat().st_size:,} bytes")

if __name__ == "__main__":
    main()
