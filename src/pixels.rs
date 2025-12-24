use ::ssimulacra2::{ColorPrimaries, TransferCharacteristic};
use pyo3::{
    FromPyObject, PyResult,
    exceptions::{PyRuntimeError, PyValueError},
};
use rayon::{
    iter::{IntoParallelRefIterator, ParallelIterator},
    slice::ParallelSlice,
};
use ssimulacra2::Rgb;

/// Quick inverse for u8 to f32 conversion
const FAST_INV: f32 = 1.0 / 255.0;

#[derive(FromPyObject)]
pub(crate) enum InputPixels {
    Flat(Vec<u8>),
    FlatF32(Vec<f32>),
    RGBTuple(Vec<[u8; 3]>),
    RGBATuple(Vec<[u8; 4]>),
    LumaTuple(Vec<[u8; 1]>),
    RGBF32TUple(Vec<[f32; 3]>),
    RGBAF32Tuple(Vec<[f32; 4]>),
    LumaF32Tuple(Vec<[f32; 1]>),
}

impl InputPixels {
    pub(crate) fn to_rgb(self, width: usize, height: usize, kind: &'static str) -> PyResult<Rgb> {
        match self {
            InputPixels::Flat(flat) => from_pixels(flat, width, height, kind),
            InputPixels::FlatF32(flat) => from_pixels_f32(flat, width, height, kind),
            InputPixels::LumaTuple(tuples) => {
                // convert to RGB by duplicating the luma value
                let pixels: Vec<[f32; 3]> = tuples
                    .par_iter()
                    .map(|chunk| {
                        let luma = chunk[0] as f32 * FAST_INV;
                        [luma, luma, luma]
                    })
                    .collect();

                let data = Rgb::new(
                    pixels,
                    width,
                    height,
                    TransferCharacteristic::SRGB,
                    ColorPrimaries::BT709,
                )
                .map_err(|err| {
                    PyRuntimeError::new_err(format!("Failed to create Rgb image: {}", err))
                })?;
                Ok(data)
            }
            InputPixels::RGBATuple(tuples) => {
                let pixels: Vec<[f32; 3]> = tuples
                    .par_iter()
                    .map(|chunk| {
                        [
                            chunk[0] as f32 * FAST_INV,
                            chunk[1] as f32 * FAST_INV,
                            chunk[2] as f32 * FAST_INV,
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
                .map_err(|err| {
                    PyRuntimeError::new_err(format!("Failed to create Rgb image: {}", err))
                })?;

                Ok(data)
            }
            InputPixels::RGBTuple(tuples) => {
                let pixels: Vec<[f32; 3]> = tuples
                    .par_iter()
                    .map(|chunk| {
                        [
                            chunk[0] as f32 * FAST_INV,
                            chunk[1] as f32 * FAST_INV,
                            chunk[2] as f32 * FAST_INV,
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
                .map_err(|err| {
                    PyRuntimeError::new_err(format!("Failed to create Rgb image: {}", err))
                })?;

                Ok(data)
            }
            InputPixels::LumaF32Tuple(tuples) => {
                // convert to RGB by duplicating the luma value
                PrecheckWrapF32::Luma(&tuples).check_pixels(kind)?;
                let pixels: Vec<[f32; 3]> = tuples
                    .par_iter()
                    .map(|chunk| {
                        let luma = chunk[0];
                        [luma, luma, luma]
                    })
                    .collect();

                let data = Rgb::new(
                    pixels,
                    width,
                    height,
                    TransferCharacteristic::SRGB,
                    ColorPrimaries::BT709,
                )
                .map_err(|err| {
                    PyRuntimeError::new_err(format!("Failed to create Rgb image: {}", err))
                })?;

                Ok(data)
            }
            InputPixels::RGBAF32Tuple(tuples) => {
                PrecheckWrapF32::RGBA(&tuples).check_pixels(kind)?;
                let pixels: Vec<[f32; 3]> = tuples
                    .par_iter()
                    .map(|chunk| [chunk[0], chunk[1], chunk[2]])
                    .collect();

                let data = Rgb::new(
                    pixels,
                    width,
                    height,
                    TransferCharacteristic::SRGB,
                    ColorPrimaries::BT709,
                )
                .map_err(|err| {
                    PyRuntimeError::new_err(format!("Failed to create Rgb image: {}", err))
                })?;

                Ok(data)
            }
            InputPixels::RGBF32TUple(tuples) => {
                PrecheckWrapF32::RGB(&tuples).check_pixels(kind)?;
                let data = Rgb::new(
                    tuples,
                    width,
                    height,
                    TransferCharacteristic::SRGB,
                    ColorPrimaries::BT709,
                )
                .map_err(|err| {
                    PyRuntimeError::new_err(format!("Failed to create Rgb image: {}", err))
                })?;

                Ok(data)
            }
        }
    }
}

fn from_pixels_rgba(data: Vec<u8>, width: usize, height: usize) -> PyResult<Rgb> {
    // We already verified the length in precheck
    let pixels: Vec<[f32; 3]> = data
        .par_chunks_exact(4)
        .map(|chunk| {
            [
                chunk[0] as f32 * FAST_INV,
                chunk[1] as f32 * FAST_INV,
                chunk[2] as f32 * FAST_INV,
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

fn from_pixels_rgb(data: Vec<u8>, width: usize, height: usize) -> PyResult<Rgb> {
    // We already verified the length in precheck
    let pixels: Vec<[f32; 3]> = data
        .par_chunks_exact(3)
        .map(|chunk| {
            [
                chunk[0] as f32 * FAST_INV,
                chunk[1] as f32 * FAST_INV,
                chunk[2] as f32 * FAST_INV,
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

fn from_pixels_luma(data: Vec<u8>, width: usize, height: usize) -> PyResult<Rgb> {
    // We already verified the length in precheck
    let pixels: Vec<[f32; 3]> = data
        .par_iter()
        .map(|chunk| {
            let luma = *chunk as f32 * FAST_INV;
            [luma, luma, luma]
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

fn from_pixels(data: Vec<u8>, width: usize, height: usize, kind: &'static str) -> PyResult<Rgb> {
    let length = data.len();
    // start by checking for RGBA
    if length == width * height * 4 {
        from_pixels_rgba(data, width, height)
    } else if length == width * height * 3 {
        from_pixels_rgb(data, width, height)
    } else if length == width * height {
        from_pixels_luma(data, width, height)
    } else {
        Err(PyValueError::new_err(format!(
            "`{kind}` length does not match width and height for any supported format (RGB, RGBA, Luma)."
        )))
    }
}

fn from_pixels_rgba_f32(data: Vec<f32>, width: usize, height: usize) -> PyResult<Rgb> {
    // We already verified the length in precheck
    let pixels: Vec<[f32; 3]> = data
        .par_chunks_exact(4)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
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

fn from_pixels_rgb_f32(data: Vec<f32>, width: usize, height: usize) -> PyResult<Rgb> {
    // We already verified the length in precheck
    let pixels: Vec<[f32; 3]> = data
        .par_chunks_exact(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
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

fn from_pixels_luma_f32(data: Vec<f32>, width: usize, height: usize) -> PyResult<Rgb> {
    // We already verified the length in precheck
    let pixels: Vec<[f32; 3]> = data
        .par_iter()
        .map(|chunk| {
            let luma = *chunk;
            [luma, luma, luma]
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

fn from_pixels_f32(
    data: Vec<f32>,
    width: usize,
    height: usize,
    kind: &'static str,
) -> PyResult<Rgb> {
    PrecheckWrapF32::Slice(&data).check_pixels(kind)?;
    let length = data.len();
    // start by checking for RGBA
    if length == width * height * 4 {
        from_pixels_rgba_f32(data, width, height)
    } else if length == width * height * 3 {
        from_pixels_rgb_f32(data, width, height)
    } else if length == width * height {
        from_pixels_luma_f32(data, width, height)
    } else {
        Err(PyValueError::new_err(format!(
            "`{kind}` length does not match width and height for any supported format (RGB, RGBA, Luma)."
        )))
    }
}

enum PrecheckWrapF32<'a> {
    Slice(&'a [f32]),
    RGB(&'a [[f32; 3]]),
    RGBA(&'a [[f32; 4]]),
    Luma(&'a [[f32; 1]]),
}

impl<'a> PrecheckWrapF32<'a> {
    fn check_pixels(self, kind: &'static str) -> PyResult<()> {
        match self {
            PrecheckWrapF32::Slice(slice) => {
                if slice.par_iter().any(outside_range_f32) {
                    return Err(PyValueError::new_err(format!(
                        "`{kind}` pixel data contains non-finite or out-of-range values (should be in [0.0, 1.0])."
                    )));
                }
                Ok(())
            }
            PrecheckWrapF32::RGB(tuples) => {
                if tuples.par_iter().any(|t| t.iter().any(outside_range_f32)) {
                    return Err(PyValueError::new_err(format!(
                        "`{kind}` pixel data contains non-finite or out-of-range values (should be in [0.0, 1.0])."
                    )));
                }
                Ok(())
            }
            PrecheckWrapF32::RGBA(tuples) => {
                if tuples.par_iter().any(|t| t.iter().any(outside_range_f32)) {
                    return Err(PyValueError::new_err(format!(
                        "`{kind}` pixel data contains non-finite or out-of-range values (should be in [0.0, 1.0])."
                    )));
                }
                Ok(())
            }
            PrecheckWrapF32::Luma(tuples) => {
                if tuples.par_iter().any(|t| outside_range_f32(&t[0])) {
                    return Err(PyValueError::new_err(format!(
                        "`{kind}` pixel data contains non-finite or out-of-range values (should be in [0.0, 1.0])."
                    )));
                }
                Ok(())
            }
        }
    }
}

fn outside_range_f32(value: &f32) -> bool {
    !value.is_finite() || *value < 0.0 || *value > 1.0
}
