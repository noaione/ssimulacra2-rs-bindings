use pyo3::{exceptions::PyRuntimeError, prelude::*};

use crate::{pixels::InputPixels, vendor::ssimulacra2::compute_frame_ssimulacra2};
mod pixels;
mod vendor;

/// ssimulacra2
/// ~~~~~~~~~~~
/// A Python binding for the SSIMULACRA2 image quality assessment algorithm rust re-implementation.
///
/// :copyright: (c) 2025 noaione
/// :license: BSD-3-Clause, see LICENSE for details.
#[pymodule(gil_used = false)]
fn _ssimulacra2(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add_function(wrap_pyfunction!(analyze, m)?)?;

    Ok(())
}

/// Analyze the given source and degraded images.
///
/// :param source: A list of integers representing the source image pixels in RGB8 format.
/// :param degraded: A list of integers representing the degraded image pixels in RGB8 format.
/// :param width: The width of the images.
/// :param height: The height of the images.
/// :return: The SSIMULACRA2 score as a float.
#[pyfunction]
#[pyo3(
    signature = (*, source, degraded, width, height)
)]
fn analyze(
    source: InputPixels,
    degraded: InputPixels,
    width: usize,
    height: usize,
) -> PyResult<f64> {
    println!("Analyzing images of size {}x{}", width, height);
    let source_rgb = source.into_rgb(width, height, "source")?;
    println!("Converted source image to RGB format");
    let degraded_rgb = degraded.into_rgb(width, height, "degraded")?;
    println!("Converted degraded image to RGB format");

    let start = std::time::Instant::now();
    let result = compute_frame_ssimulacra2(source_rgb, degraded_rgb).map_err(|err| {
        PyRuntimeError::new_err(format!("Failed to compute SSIMULACRA2: {}", err))
    })?;
    let duration = start.elapsed();
    println!("Computed SSIMULACRA2 score: {} in {:?}", result, duration);

    Ok(result)
}
