# ssimulacra2-rs

Python bindings for the [ssimulacra2](https://github.com/rust-av/ssimulacra2) re-implementation in Rust

[![PyPI version](https://img.shields.io/pypi/v/ssimulacra2-rs.svg)](https://pypi.org/project/ssimulacra2-rs/)

The following `ssimulacra2` version is used: `v0.5.1`

## Usage

`ssimulacra2-rs` can be installed from PyPI:

```bash
pip install ssimulacra2-rs
```

And can be used as follows:

```python
import ssimulacra2
from PIL import Image

source = Image.open("source_image.png")
degraded = Image.open("degraded_image.png")

## We support the following format input:
# - RGB8: mode "RGB", 8 bits per channel
# - RGBF32: mode "RGB", 32 bits float per channel
# - RGBA8: mode "RGBA", 8 bits per channel (alpha channel will be ignored)
# - RGBAF32: mode "RGBA", 32 bits float per channel (alpha channel will be ignored)
# - L8: mode "L", 8 bits grayscale (will be converted to RGB by replicating the channel)
# - L32F: mode "F", 32 bits float grayscale (will be converted to RGB by replicating the channel)
source = source.convert("RGB")
degraded = degraded.convert("RGB")

# In `RGB`/`RGBA`, the pixel data would be a list of (R, G, B, [A]) tuples
# While `L` mode would be a list of single values
source_pixels = list(source.getdata())
degraded_pixels = list(degraded.getdata())

# Pass the pixel data along with image width and height (recommended from the `source` image)
assessment = ssimulacra2.analyze(source_pixels, degraded_pixels, source.width, source.height)
# Return a float value representing the SSIMULACRA2 score (`f64` in Rust)
print(f"SSIMULACRA2 score: {assessment:.2f}")
```

## License

This project is licensed under the **BSD-3-Clause License**. See the [LICENSE](https://github.com/noaione/ssimulacra2-rs-bindings/blob/master/LICENSE) file for details.

### Acknowledgements
The project use the following library:
- [ssimulacra2-rs](https://github.com/rust-av/ssimulacra2) which is licensed under the BSD-2-Clause License.
- [ssimulacra2](https://github.com/cloudinary/ssimulacra2), the original implementation of SSIMULACRA2 in C/C++ by Cloudinary.
