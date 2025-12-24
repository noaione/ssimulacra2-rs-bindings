# ssimulacra2-rs

Python bindings for the [ssimulacra2](https://github.com/rust-av/ssimulacra2) re-implementation in Rust

[![PyPI version](https://img.shields.io/pypi/v/ssimulacra2-rs.svg)](https://pypi.org/project/resdet/)

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

source = source.convert("RGB") # Only RGB images are supported
degraded = degraded.convert("RGB")

source_pixels = list(source.getdata())
degraded_pixels = list(degraded.getdata())

source_pixels = [value for pixel in source_pixels for value in pixel]
degraded_pixels = [value for pixel in degraded_pixels for value in pixel]

assessment = ssimulacra2.analyze(source_pixels, degraded_pixels, source.width, source.height)
print(f"SSIMULACRA2 score: {assessment:.2f}")
```

## License

This project is licensed under the BSD-3-Clause License. See the [LICENSE](LICENSE) file for details.

The project also use library from [ssimulacra2](https://github.com/rust-av/ssimulacra2) which is licensed under the BSD-2-Clause License.