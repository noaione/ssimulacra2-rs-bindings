use std::os::raw::{c_int, c_uchar, c_uint};

const SSIMU2_OK: i32 = 0;
const SSIMU2_INVALID_CHANNELS: i32 = 1;
const SSIMU2_OUT_OF_MEMORY: i32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ComputeError {
    #[error("Invalid number of channels provided")]
    InvalidChannels,
    #[error("Out of memory")]
    OutOfMemory,
    #[error("Unknown error occurred, code: {0}")]
    Unknown(i32),
}

unsafe extern "C" {
    fn ssimulacra2_score(
        reference: *const c_uchar, // uint8_t*
        distorted: *const c_uchar, // uint8_t*
        width: c_uint,             // unsigned int
        height: c_uint,            // unsigned int
        channels: c_uint,          // unsigned int
        out_score: *mut f64,       // double*
    ) -> c_int;
}

pub fn compute_frame_ssimulacra2(
    reference: Vec<u8>,
    distorted: Vec<u8>,
    width: usize,
    height: usize,
) -> Result<f64, ComputeError> {
    let mut score: f64 = 0.0;

    // make reference and distored to pointer

    let status = unsafe {
        ssimulacra2_score(
            reference.as_ptr(),
            distorted.as_ptr(),
            width as c_uint,
            height as c_uint,
            // internally we do RGB only
            3, // channels
            &mut score as *mut f64,
        )
    };

    match status {
        SSIMU2_OK => Ok(score),
        SSIMU2_INVALID_CHANNELS => Err(ComputeError::InvalidChannels),
        SSIMU2_OUT_OF_MEMORY => Err(ComputeError::OutOfMemory),
        other => Err(ComputeError::Unknown(other)),
    }
}
