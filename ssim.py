from pathlib import Path
import time
import ssimulacra2
from PIL import Image

first_img = Path(r"x/I Want to Love You Till Your Dying Day - c025 (v06) - p005 [dig] [Thank You] [Kodansha Comics] [nao] {HQ}.png")
degraded_img = Path(r"y/I Want to Love You Till Your Dying Day - c025 (v06) - p005 [dig] [Thank You] [Kodansha Comics] [nao] {HQ}.png")

with Image.open(first_img) as src_img, Image.open(degraded_img) as deg_img:
    perf_start = time.perf_counter()
    score = ssimulacra2.pil_analyze(
        source=src_img,
        degraded=deg_img,
    )
    perf_end = time.perf_counter()
    print(f"SSIMULACRA2 score: {score:.2f}")
    print(f"Time taken: {perf_end - perf_start:.4f} seconds")
