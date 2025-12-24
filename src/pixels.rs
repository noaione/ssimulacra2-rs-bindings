use pyo3::{FromPyObject, PyResult, exceptions::PyValueError};
use rayon::{
    iter::{IntoParallelRefIterator, ParallelIterator},
    slice::ParallelSlice,
};

/// Simple RGB image representation
type Rgb = Vec<u8>;

#[derive(FromPyObject)]
pub(crate) enum InputPixels {
    Flat(Vec<u8>),
    RGBTuple(Vec<[u8; 3]>),
    RGBATuple(Vec<[u8; 4]>),
    LumaTuple(Vec<[u8; 1]>),
}

impl InputPixels {
    pub(crate) fn into_flat_rgb(
        self,
        width: usize,
        height: usize,
        kind: &'static str,
    ) -> PyResult<Rgb> {
        match self {
            InputPixels::Flat(flat) => from_pixels(flat, width, height, kind),
            InputPixels::LumaTuple(tuples) => {
                let pixels: Rgb = tuples
                    .par_iter()
                    .map(|luma| vec![luma[0]; 3])
                    .flatten()
                    .collect();
                Ok(pixels)
            }
            InputPixels::RGBATuple(tuples) => {
                let pixels: Rgb = tuples
                    .par_iter()
                    .map(|rgba| vec![rgba[0], rgba[1], rgba[2]])
                    .flatten()
                    .collect();
                Ok(pixels)
            }
            InputPixels::RGBTuple(tuples) => {
                let pixels: Rgb = tuples
                    .par_iter()
                    .map(|rgb| vec![rgb[0], rgb[1], rgb[2]])
                    .flatten()
                    .collect();

                Ok(pixels)
            }
        }
    }
}

fn from_pixels_rgba(data: Vec<u8>) -> PyResult<Rgb> {
    // We already verified the length in precheck
    let pixels: Rgb = data
        .par_chunks_exact(4)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .flatten()
        .collect();

    Ok(pixels)
}

fn from_pixels_rgb(data: Vec<u8>) -> PyResult<Rgb> {
    // We already verified the length in precheck
    let pixels: Rgb = data
        .par_chunks_exact(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .flatten()
        .collect();

    Ok(pixels)
}

fn from_pixels_luma(data: Vec<u8>) -> PyResult<Rgb> {
    // We already verified the length in precheck
    let pixels: Rgb = data
        .par_iter()
        .map(|chunk| {
            let luma = *chunk;
            [luma, luma, luma]
        })
        .flatten()
        .collect();

    Ok(pixels)
}

fn from_pixels(data: Vec<u8>, width: usize, height: usize, kind: &'static str) -> PyResult<Rgb> {
    let length = data.len();
    // start by checking for RGBA
    if length == width * height * 4 {
        from_pixels_rgba(data)
    } else if length == width * height * 3 {
        from_pixels_rgb(data)
    } else if length == width * height {
        from_pixels_luma(data)
    } else {
        Err(PyValueError::new_err(format!(
            "`{kind}` length does not match width and height for any supported format (RGB, RGBA, Luma)."
        )))
    }
}
