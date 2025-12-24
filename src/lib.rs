use pyo3::{exceptions::PyRuntimeError, prelude::*};

use crate::{bindings::compute_frame_ssimulacra2, pixels::InputPixels};
mod bindings;
mod pixels;

/// ssimulacra2
/// ~~~~~~~~~~~
/// A Python binding for the SSIMULACRA2 image quality assessment algorithm Zig re-implementation.
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
    let source_rgb = source.into_flat_rgb(width, height, "source")?;
    let degraded_rgb = degraded.into_flat_rgb(width, height, "degraded")?;

    let result =
        compute_frame_ssimulacra2(source_rgb, degraded_rgb, width, height).map_err(|err| {
            PyRuntimeError::new_err(format!("Failed to compute SSIMULACRA2: {}", err))
        })?;

    Ok(result)
}
