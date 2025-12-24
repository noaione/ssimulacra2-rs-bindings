use ::ssimulacra2::{ColorPrimaries, Rgb, TransferCharacteristic, compute_frame_ssimulacra2};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use rayon::{iter::ParallelIterator, slice::ParallelSlice};

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
fn analyze(source: Vec<u8>, degraded: Vec<u8>, width: usize, height: usize) -> PyResult<f64> {
    precheck(&source, width, height, "source")?;
    precheck(&degraded, width, height, "degraded")?;

    let source_rgb = from_pixels(source, width, height)?;
    let distorted_rgb = from_pixels(degraded, width, height)?;
    let result = compute_frame_ssimulacra2(source_rgb, distorted_rgb).map_err(|err| {
        PyRuntimeError::new_err(format!("Failed to compute SSIMULACRA2: {}", err))
    })?;

    Ok(result)
}

fn from_pixels(data: Vec<u8>, width: usize, height: usize) -> PyResult<Rgb> {
    // We already verified the length in precheck
    let fast_inv: f32 = 1.0 / 255.0;
    let pixels: Vec<[f32; 3]> = data
        .par_chunks_exact(3)
        .map(|chunk| {
            [
                chunk[0] as f32 * fast_inv,
                chunk[1] as f32 * fast_inv,
                chunk[2] as f32 * fast_inv,
            ]
        })
        .collect();

    let data = Rgb::new(
        pixels,
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .map_err(|err| PyRuntimeError::new_err(format!("Failed to create Rgb image: {}", err)))?;

    Ok(data)
}

fn precheck(data: &[u8], width: usize, height: usize, kind: &'static str) -> PyResult<()> {
    if data.len() != width * height * 3 {
        return Err(PyValueError::new_err(format!(
            "{kind} length does not match width and height: expected {}, got {}",
            width * height * 3,
            data.len()
        )));
    }

    Ok(())
}
