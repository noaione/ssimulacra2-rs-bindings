use gaussian::RecursiveGaussian;
pub(crate) use gaussian::SimdBackend;

/// Structure handling image blur.
///
/// This struct contains the necessary buffers and the kernel used for blurring
/// (currently a recursive approximation of the Gaussian filter).
///
/// Note that the width and height of the image passed to [blur][Self::blur] needs to exactly
/// match the width and height of this instance. If you reduce the image size (e.g. via
/// downscaling), [`shrink_to`][Self::shrink_to] can be used to resize the internal buffers.
#[allow(dead_code)]
pub struct Blur {
    kernel: RecursiveGaussian,
    temp: Vec<f32>,
    width: usize,
    height: usize,
}

/// SIMD-optimized variant of [`Blur`].
///
/// This keeps the same buffer ownership model as `Blur` (re-using allocations via `shrink_to`),
/// but routes the horizontal+vertical passes through SIMD-enabled kernels.
pub struct BlurSimd {
    kernel: RecursiveGaussian,
    temp: Vec<f32>,
    width: usize,
    height: usize,
    backend: gaussian::SimdBackend,
}

#[allow(dead_code)]
impl Blur {
    /// Create a new [Blur] for images of the given width and height.
    /// This pre-allocates the necessary buffers.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Blur {
            kernel: RecursiveGaussian,
            temp: vec![0.0f32; width * height],
            width,
            height,
        }
    }

    /// Truncates the internal buffers to fit images of the given width and height.
    ///
    /// This will [truncate][Vec::truncate] the internal buffers
    /// without affecting the allocated memory.
    pub fn shrink_to(&mut self, width: usize, height: usize) {
        self.temp.truncate(width * height);
        self.width = width;
        self.height = height;
    }

    /// Blur the given image.
    pub fn blur(&mut self, img: &[Vec<f32>; 3]) -> [Vec<f32>; 3] {
        [
            self.blur_plane(&img[0]),
            self.blur_plane(&img[1]),
            self.blur_plane(&img[2]),
        ]
    }

    fn blur_plane(&mut self, plane: &[f32]) -> Vec<f32> {
        let mut out = vec![0f32; self.width * self.height];
        self.kernel
            .horizontal_pass(plane, &mut self.temp, self.width);
        self.kernel
            .vertical_pass_chunked::<128, 32>(&self.temp, &mut out, self.width, self.height);
        out
    }
}

impl BlurSimd {
    /// Create a new [`BlurSimd`] for images of the given width and height.
    /// This pre-allocates the necessary buffers.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        BlurSimd {
            kernel: RecursiveGaussian,
            temp: vec![0.0f32; width * height],
            width,
            height,
            backend: gaussian::SimdBackend::detect(),
        }
    }

    /// Get the detected SIMD backend
    pub fn backend(&self) -> gaussian::SimdBackend {
        self.backend
    }

    /// Set the SIMD backend to use
    pub fn set_backend(&mut self, backend: gaussian::SimdBackend) {
        self.backend = backend;
    }

    /// Truncates the internal buffers to fit images of the given width and height.
    ///
    /// This will [truncate][Vec::truncate] the internal buffers
    /// without affecting the allocated memory.
    pub fn shrink_to(&mut self, width: usize, height: usize) {
        self.temp.truncate(width * height);
        self.width = width;
        self.height = height;
    }

    /// Blur the given image.
    pub fn blur(&mut self, img: &[Vec<f32>; 3]) -> [Vec<f32>; 3] {
        [
            self.blur_plane(&img[0]),
            self.blur_plane(&img[1]),
            self.blur_plane(&img[2]),
        ]
    }

    fn blur_plane(&mut self, plane: &[f32]) -> Vec<f32> {
        let mut out = vec![0f32; self.width * self.height];
        self.kernel.horizontal_pass_simd_with_backend(
            self.backend,
            plane,
            &mut self.temp,
            self.width,
        );
        self.kernel.vertical_pass_simd_with_backend(
            self.backend,
            &self.temp,
            &mut out,
            self.width,
            self.height,
        );
        out
    }
}

mod gaussian {
    mod consts {
        #![allow(clippy::unreadable_literal)]
        include!(concat!(env!("OUT_DIR"), "/recursive_gaussian.rs"));
    }

    /// Implements "Recursive Implementation of the Gaussian Filter Using Truncated
    /// Cosine Functions" by Charalampidis [2016].
    pub struct RecursiveGaussian;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum SimdBackend {
        Scalar,
        #[cfg(target_arch = "x86_64")]
        Sse42,
        #[cfg(target_arch = "x86_64")]
        Avx,
        #[cfg(target_arch = "x86_64")]
        Avx512,
        #[cfg(target_arch = "aarch64")]
        Neon,
    }

    impl SimdBackend {
        #[must_use]
        pub(crate) fn detect() -> Self {
            use jxl_simd::SimdDescriptor;

            #[cfg(target_arch = "x86_64")]
            {
                if jxl_simd::Avx512Descriptor::new().is_some() {
                    return Self::Avx512;
                }
                if jxl_simd::AvxDescriptor::new().is_some() {
                    return Self::Avx;
                }
                if jxl_simd::Sse42Descriptor::new().is_some() {
                    return Self::Sse42;
                }
            }

            #[cfg(target_arch = "aarch64")]
            {
                if jxl_simd::NeonDescriptor::new().is_some() {
                    return Self::Neon;
                }
            }

            Self::Scalar
        }
    }

    impl RecursiveGaussian {
        #[allow(dead_code)]
        pub fn horizontal_pass(&self, input: &[f32], output: &mut [f32], width: usize) {
            use rayon::iter::{IndexedParallelIterator, ParallelIterator};
            use rayon::prelude::ParallelSliceMut;
            use rayon::slice::ParallelSlice;

            assert_eq!(input.len(), output.len());

            input
                .par_chunks_exact(width)
                .zip(output.par_chunks_exact_mut(width))
                .for_each(|(input, output)| self.horizontal_row(input, output, width));
        }

        /// SIMD-accelerated variant of [`horizontal_pass`][Self::horizontal_pass].
        ///
        /// This is a recursive (IIR) filter horizontally, which cannot be vectorized across X
        /// within a single row due to data dependencies.
        ///
        /// Instead, this processes multiple rows at once: each SIMD lane holds the state for a
        /// different row, and we iterate across X in lockstep.
        #[allow(dead_code)]
        pub fn horizontal_pass_simd(&self, input: &[f32], output: &mut [f32], width: usize) {
            self.horizontal_pass_simd_with_backend(SimdBackend::detect(), input, output, width);
        }

        pub fn horizontal_pass_simd_with_backend(
            &self,
            backend: SimdBackend,
            input: &[f32],
            output: &mut [f32],
            width: usize,
        ) {
            use jxl_simd::SimdDescriptor;

            match backend {
                #[cfg(target_arch = "x86_64")]
                SimdBackend::Avx512 => unsafe { jxl_simd::Avx512Descriptor::new_unchecked() }
                    .call(|d| self.horizontal_pass_simd_impl(d, input, output, width)),
                #[cfg(target_arch = "x86_64")]
                SimdBackend::Avx => unsafe { jxl_simd::AvxDescriptor::new_unchecked() }
                    .call(|d| self.horizontal_pass_simd_impl(d, input, output, width)),
                #[cfg(target_arch = "x86_64")]
                SimdBackend::Sse42 => unsafe { jxl_simd::Sse42Descriptor::new_unchecked() }
                    .call(|d| self.horizontal_pass_simd_impl(d, input, output, width)),
                #[cfg(target_arch = "aarch64")]
                SimdBackend::Neon => unsafe { jxl_simd::NeonDescriptor::new_unchecked() }
                    .call(|d| self.horizontal_pass_simd_impl(d, input, output, width)),
                SimdBackend::Scalar => jxl_simd::ScalarDescriptor
                    .call(|d| self.horizontal_pass_simd_impl(d, input, output, width)),
            }
        }

        fn horizontal_pass_simd_impl<D: jxl_simd::SimdDescriptor>(
            &self,
            d: D,
            input: &[f32],
            output: &mut [f32],
            width: usize,
        ) {
            use jxl_simd::F32SimdVec;
            use rayon::iter::{IndexedParallelIterator, ParallelIterator};
            use rayon::prelude::ParallelSliceMut;
            use rayon::slice::ParallelSlice;

            assert_eq!(input.len(), output.len());
            assert!(width > 0);
            assert_eq!(input.len() % width, 0);

            let lanes = D::F32Vec::LEN;
            let rows = input.len() / width;
            let simd_blocks = rows / lanes;

            // Process blocks of `lanes` rows in parallel.
            input
                .par_chunks_exact(width * lanes)
                .zip(output.par_chunks_exact_mut(width * lanes))
                .take(simd_blocks)
                .for_each(|(input_block, output_block)| {
                    self.horizontal_rows_simd(d, input_block, output_block, width)
                });

            // Remainder rows: fall back to scalar.
            let rem_start = simd_blocks * width * lanes;
            if rem_start < input.len() {
                let input_rem = &input[rem_start..];
                let output_rem = &mut output[rem_start..];
                input_rem
                    .par_chunks_exact(width)
                    .zip(output_rem.par_chunks_exact_mut(width))
                    .for_each(|(input_row, output_row)| {
                        self.horizontal_row(input_row, output_row, width)
                    });
            }
        }

        fn horizontal_rows_simd<D: jxl_simd::SimdDescriptor>(
            &self,
            d: D,
            input: &[f32],
            output: &mut [f32],
            width: usize,
        ) {
            use jxl_simd::F32SimdVec;

            let lanes = D::F32Vec::LEN;
            debug_assert!(lanes <= 16);
            debug_assert_eq!(input.len(), width * lanes);
            debug_assert_eq!(output.len(), width * lanes);

            let big_n = consts::RADIUS as isize;

            let mut prev_1 = D::F32Vec::zero(d);
            let mut prev_3 = D::F32Vec::zero(d);
            let mut prev_5 = D::F32Vec::zero(d);
            let mut prev2_1 = D::F32Vec::zero(d);
            let mut prev2_3 = D::F32Vec::zero(d);
            let mut prev2_5 = D::F32Vec::zero(d);

            let mul_in_1 = D::F32Vec::splat(d, consts::MUL_IN_1);
            let mul_in_3 = D::F32Vec::splat(d, consts::MUL_IN_3);
            let mul_in_5 = D::F32Vec::splat(d, consts::MUL_IN_5);

            let mul_prev2_1 = D::F32Vec::splat(d, consts::MUL_PREV2_1);
            let mul_prev2_3 = D::F32Vec::splat(d, consts::MUL_PREV2_3);
            let mul_prev2_5 = D::F32Vec::splat(d, consts::MUL_PREV2_5);

            let mul_prev_1 = D::F32Vec::splat(d, consts::MUL_PREV_1);
            let mul_prev_3 = D::F32Vec::splat(d, consts::MUL_PREV_3);
            let mul_prev_5 = D::F32Vec::splat(d, consts::MUL_PREV_5);

            let input_ptr = input.as_ptr();
            let output_ptr = output.as_mut_ptr();

            let mut n = (-big_n) + 1;
            while n < width as isize {
                let left = n - big_n - 1;
                let right = n + big_n - 1;

                let left_vec = if left >= 0 {
                    let idx = left as usize;
                    let mut a = [0.0f32; 16];
                    let a_slice = &mut a[..lanes];
                    for lane in 0..lanes {
                        // SAFETY: idx < width; lane*width+idx in-bounds for this block.
                        unsafe { a_slice[lane] = *input_ptr.add(lane * width + idx) };
                    }
                    D::F32Vec::load(d, a_slice)
                } else {
                    D::F32Vec::zero(d)
                };

                let right_vec = if right < width as isize {
                    let idx = right as usize;
                    let mut a = [0.0f32; 16];
                    let a_slice = &mut a[..lanes];
                    for lane in 0..lanes {
                        unsafe { a_slice[lane] = *input_ptr.add(lane * width + idx) };
                    }
                    D::F32Vec::load(d, a_slice)
                } else {
                    D::F32Vec::zero(d)
                };

                let sum = left_vec + right_vec;

                let mut out_1 = sum * mul_in_1;
                let mut out_3 = sum * mul_in_3;
                let mut out_5 = sum * mul_in_5;

                out_1 = mul_prev2_1.mul_add(prev2_1, out_1);
                out_3 = mul_prev2_3.mul_add(prev2_3, out_3);
                out_5 = mul_prev2_5.mul_add(prev2_5, out_5);
                prev2_1 = prev_1;
                prev2_3 = prev_3;
                prev2_5 = prev_5;

                out_1 = mul_prev_1.mul_add(prev_1, out_1);
                out_3 = mul_prev_3.mul_add(prev_3, out_3);
                out_5 = mul_prev_5.mul_add(prev_5, out_5);
                prev_1 = out_1;
                prev_3 = out_3;
                prev_5 = out_5;

                if n >= 0 {
                    let x = n as usize;
                    let total = out_1 + out_3 + out_5;
                    let mut tmp = [0.0f32; 16];
                    let tmp_slice = &mut tmp[..lanes];
                    total.store(tmp_slice);
                    for lane in 0..lanes {
                        // SAFETY: x < width; lane*width+x in-bounds for this output block.
                        unsafe {
                            *output_ptr.add(lane * width + x) = tmp_slice[lane];
                        }
                    }
                }

                n += 1;
            }
        }

        fn horizontal_row(&self, input: &[f32], output: &mut [f32], width: usize) {
            let big_n = consts::RADIUS as isize;
            let mut prev_1 = 0f32;
            let mut prev_3 = 0f32;
            let mut prev_5 = 0f32;
            let mut prev2_1 = 0f32;
            let mut prev2_3 = 0f32;
            let mut prev2_5 = 0f32;

            let mut n = (-big_n) + 1;
            while n < width as isize {
                let left = n - big_n - 1;
                let right = n + big_n - 1;
                let left_val = if left >= 0 {
                    // SAFETY: `left` can never be bigger than `width`
                    unsafe { *input.get_unchecked(left as usize) }
                } else {
                    0f32
                };
                let right_val = if right < width as isize {
                    // SAFETY: this branch ensures that `right` is not bigger than `width`
                    unsafe { *input.get_unchecked(right as usize) }
                } else {
                    0f32
                };
                let sum = left_val + right_val;

                let mut out_1 = sum * consts::MUL_IN_1;
                let mut out_3 = sum * consts::MUL_IN_3;
                let mut out_5 = sum * consts::MUL_IN_5;

                out_1 = consts::MUL_PREV2_1.mul_add(prev2_1, out_1);
                out_3 = consts::MUL_PREV2_3.mul_add(prev2_3, out_3);
                out_5 = consts::MUL_PREV2_5.mul_add(prev2_5, out_5);
                prev2_1 = prev_1;
                prev2_3 = prev_3;
                prev2_5 = prev_5;

                out_1 = consts::MUL_PREV_1.mul_add(prev_1, out_1);
                out_3 = consts::MUL_PREV_3.mul_add(prev_3, out_3);
                out_5 = consts::MUL_PREV_5.mul_add(prev_5, out_5);
                prev_1 = out_1;
                prev_3 = out_3;
                prev_5 = out_5;

                if n >= 0 {
                    // SAFETY: We know that this chunk of output is of size `width`,
                    // which `n` cannot be larger than.
                    unsafe {
                        *output.get_unchecked_mut(n as usize) = out_1 + out_3 + out_5;
                    }
                }

                n += 1;
            }
        }

        /// SIMD-accelerated vertical pass.
        ///
        /// This is a recursive (IIR) filter vertically, which cannot be vectorized across Y
        /// within a single column due to data dependencies.
        ///
        /// Instead, this processes multiple columns at once: each SIMD lane holds the state for
        /// a different column, and we iterate across Y in lockstep.
        #[allow(dead_code)]
        pub fn vertical_pass_simd(
            &self,
            input: &[f32],
            output: &mut [f32],
            width: usize,
            height: usize,
        ) {
            self.vertical_pass_simd_with_backend(
                SimdBackend::detect(),
                input,
                output,
                width,
                height,
            );
        }

        pub fn vertical_pass_simd_with_backend(
            &self,
            backend: SimdBackend,
            input: &[f32],
            output: &mut [f32],
            width: usize,
            height: usize,
        ) {
            use jxl_simd::SimdDescriptor;

            assert_eq!(input.len(), output.len());
            assert_eq!(input.len(), width * height);
            assert!(width > 0);
            assert!(height > 0);

            match backend {
                #[cfg(target_arch = "x86_64")]
                SimdBackend::Avx512 => unsafe { jxl_simd::Avx512Descriptor::new_unchecked() }
                    .call(|d| self.vertical_pass_simd_impl(d, input, output, width, height)),
                #[cfg(target_arch = "x86_64")]
                SimdBackend::Avx => unsafe { jxl_simd::AvxDescriptor::new_unchecked() }
                    .call(|d| self.vertical_pass_simd_impl(d, input, output, width, height)),
                #[cfg(target_arch = "x86_64")]
                SimdBackend::Sse42 => unsafe { jxl_simd::Sse42Descriptor::new_unchecked() }
                    .call(|d| self.vertical_pass_simd_impl(d, input, output, width, height)),
                #[cfg(target_arch = "aarch64")]
                SimdBackend::Neon => unsafe { jxl_simd::NeonDescriptor::new_unchecked() }
                    .call(|d| self.vertical_pass_simd_impl(d, input, output, width, height)),
                SimdBackend::Scalar => jxl_simd::ScalarDescriptor
                    .call(|d| self.vertical_pass_simd_impl(d, input, output, width, height)),
            }
        }

        fn vertical_pass_simd_impl<D: jxl_simd::SimdDescriptor>(
            &self,
            d: D,
            input: &[f32],
            output: &mut [f32],
            width: usize,
            height: usize,
        ) {
            use jxl_simd::F32SimdVec;

            let lanes = D::F32Vec::LEN;
            debug_assert!(lanes <= 16);

            let mut x = 0;
            while x + lanes <= width {
                self.vertical_columns_simd::<D>(d, &input[x..], &mut output[x..], width, height);
                x += lanes;
            }

            while x < width {
                self.vertical_pass_1(&input[x..], &mut output[x..], width, height);
                x += 1;
            }
        }

        fn vertical_columns_simd<D: jxl_simd::SimdDescriptor>(
            &self,
            d: D,
            input: &[f32],
            output: &mut [f32],
            width: usize,
            height: usize,
        ) {
            use jxl_simd::F32SimdVec;

            let lanes = D::F32Vec::LEN;
            debug_assert!(lanes <= 16);

            let big_n = consts::RADIUS as isize;

            let zero = D::F32Vec::zero(d);

            let vert_mul_in_1 = D::F32Vec::splat(d, consts::VERT_MUL_IN_1);
            let vert_mul_in_3 = D::F32Vec::splat(d, consts::VERT_MUL_IN_3);
            let vert_mul_in_5 = D::F32Vec::splat(d, consts::VERT_MUL_IN_5);

            let vert_mul_prev_1 = D::F32Vec::splat(d, consts::VERT_MUL_PREV_1);
            let vert_mul_prev_3 = D::F32Vec::splat(d, consts::VERT_MUL_PREV_3);
            let vert_mul_prev_5 = D::F32Vec::splat(d, consts::VERT_MUL_PREV_5);

            let mut prev_1 = zero;
            let mut prev_3 = zero;
            let mut prev_5 = zero;
            let mut prev2_1 = zero;
            let mut prev2_3 = zero;
            let mut prev2_5 = zero;

            let input_ptr = input.as_ptr();
            let output_ptr = output.as_mut_ptr();

            let mut n = (-big_n) + 1;
            while n < height as isize {
                let top = n - big_n - 1;
                let bottom = n + big_n - 1;

                let top_vec = if top >= 0 {
                    let row_off = top as usize * width;
                    let mut a = [0.0f32; 16];
                    let a_slice = &mut a[..lanes];
                    for i in 0..lanes {
                        // SAFETY: `row_off + i` in bounds for row, and `input` starts at column x.
                        unsafe { a_slice[i] = *input_ptr.add(row_off + i) };
                    }
                    D::F32Vec::load(d, a_slice)
                } else {
                    zero
                };

                let bottom_vec = if bottom < height as isize {
                    let row_off = bottom as usize * width;
                    let mut a = [0.0f32; 16];
                    let a_slice = &mut a[..lanes];
                    for i in 0..lanes {
                        unsafe { a_slice[i] = *input_ptr.add(row_off + i) };
                    }
                    D::F32Vec::load(d, a_slice)
                } else {
                    zero
                };

                let sum = top_vec + bottom_vec;

                let t1 = prev_1.mul_add(vert_mul_prev_1, prev2_1);
                let t3 = prev_3.mul_add(vert_mul_prev_3, prev2_3);
                let t5 = prev_5.mul_add(vert_mul_prev_5, prev2_5);

                let out_1 = sum.mul_add(vert_mul_in_1, zero - t1);
                let out_3 = sum.mul_add(vert_mul_in_3, zero - t3);
                let out_5 = sum.mul_add(vert_mul_in_5, zero - t5);

                if n >= 0 {
                    let row_off = n as usize * width;
                    let total = out_1 + out_3 + out_5;
                    let mut tmp = [0.0f32; 16];
                    let tmp_slice = &mut tmp[..lanes];
                    total.store(tmp_slice);
                    for i in 0..lanes {
                        // SAFETY: write within this row; `output` starts at column x.
                        unsafe { *output_ptr.add(row_off + i) = tmp_slice[i] };
                    }
                }

                prev2_1 = prev_1;
                prev2_3 = prev_3;
                prev2_5 = prev_5;
                prev_1 = out_1;
                prev_3 = out_3;
                prev_5 = out_5;

                n += 1;
            }
        }

        pub fn vertical_pass_chunked<const J: usize, const K: usize>(
            &self,
            input: &[f32],
            output: &mut [f32],
            width: usize,
            height: usize,
        ) {
            assert!(J > K);
            assert!(K > 0);

            assert_eq!(input.len(), output.len());

            let mut x = 0;
            while x + J <= width {
                if J == 128 {
                    self.vertical_pass_128(&input[x..], &mut output[x..], width, height);
                } else if J == 32 {
                    self.vertical_pass_32(&input[x..], &mut output[x..], width, height);
                } else {
                    self.vertical_pass::<J>(&input[x..], &mut output[x..], width, height);
                }
                x += J;
            }

            while x + K <= width {
                if K == 128 {
                    self.vertical_pass_128(&input[x..], &mut output[x..], width, height);
                } else if K == 32 {
                    self.vertical_pass_32(&input[x..], &mut output[x..], width, height);
                } else {
                    self.vertical_pass::<K>(&input[x..], &mut output[x..], width, height);
                }
                x += K;
            }

            while x < width {
                self.vertical_pass_1(&input[x..], &mut output[x..], width, height);
                x += 1;
            }
        }

        fn vertical_pass_128(
            &self,
            input: &[f32],
            output: &mut [f32],
            width: usize,
            height: usize,
        ) {
            self.vertical_pass_fixed::<128, 384>(input, output, width, height);
        }

        fn vertical_pass_32(&self, input: &[f32], output: &mut [f32], width: usize, height: usize) {
            self.vertical_pass_fixed::<32, 96>(input, output, width, height);
        }

        fn vertical_pass_1(&self, input: &[f32], output: &mut [f32], width: usize, height: usize) {
            self.vertical_pass_fixed::<1, 3>(input, output, width, height);
        }

        fn vertical_pass_fixed<const COLUMNS: usize, const STATE: usize>(
            &self,
            input: &[f32],
            output: &mut [f32],
            width: usize,
            height: usize,
        ) {
            debug_assert_eq!(STATE, 3 * COLUMNS);
            assert_eq!(input.len(), output.len());

            let big_n = consts::RADIUS as isize;

            let zeroes = [0.0f32; COLUMNS];
            let mut prev = [0.0f32; STATE];
            let mut prev2 = [0.0f32; STATE];
            let mut out = [0.0f32; STATE];

            let mut n = (-big_n) + 1;
            while n < height as isize {
                let top = n - big_n - 1;
                let bottom = n + big_n - 1;
                let top_row = if top >= 0 {
                    &input[top as usize * width..][..COLUMNS]
                } else {
                    &zeroes
                };

                let bottom_row = if bottom < height as isize {
                    &input[bottom as usize * width..][..COLUMNS]
                } else {
                    &zeroes
                };

                for i in 0..COLUMNS {
                    let sum = top_row[i] + bottom_row[i];

                    let i1 = i;
                    let i3 = i1 + COLUMNS;
                    let i5 = i3 + COLUMNS;

                    let out1 = prev[i1].mul_add(consts::VERT_MUL_PREV_1, prev2[i1]);
                    let out3 = prev[i3].mul_add(consts::VERT_MUL_PREV_3, prev2[i3]);
                    let out5 = prev[i5].mul_add(consts::VERT_MUL_PREV_5, prev2[i5]);

                    let out1 = sum.mul_add(consts::VERT_MUL_IN_1, -out1);
                    let out3 = sum.mul_add(consts::VERT_MUL_IN_3, -out3);
                    let out5 = sum.mul_add(consts::VERT_MUL_IN_5, -out5);

                    out[i1] = out1;
                    out[i3] = out3;
                    out[i5] = out5;

                    if n >= 0 {
                        output[n as usize * width + i] = out1 + out3 + out5;
                    }
                }

                prev2.copy_from_slice(&prev);
                prev.copy_from_slice(&out);

                n += 1;
            }
        }

        // Apply 1D vertical scan on COLUMNS elements at a time
        #[allow(dead_code)]
        pub fn vertical_pass<const COLUMNS: usize>(
            &self,
            input: &[f32],
            output: &mut [f32],
            width: usize,
            height: usize,
        ) {
            assert_eq!(input.len(), output.len());

            let big_n = consts::RADIUS as isize;

            let zeroes = vec![0f32; COLUMNS];
            let mut prev = vec![0f32; 3 * COLUMNS];
            let mut prev2 = vec![0f32; 3 * COLUMNS];
            let mut out = vec![0f32; 3 * COLUMNS];

            let mut n = (-big_n) + 1;
            while n < height as isize {
                let top = n - big_n - 1;
                let bottom = n + big_n - 1;
                let top_row = if top >= 0 {
                    &input[top as usize * width..][..COLUMNS]
                } else {
                    &zeroes
                };

                let bottom_row = if bottom < height as isize {
                    &input[bottom as usize * width..][..COLUMNS]
                } else {
                    &zeroes
                };

                for i in 0..COLUMNS {
                    let sum = top_row[i] + bottom_row[i];

                    let i1 = i;
                    let i3 = i1 + COLUMNS;
                    let i5 = i3 + COLUMNS;

                    let out1 = prev[i1].mul_add(consts::VERT_MUL_PREV_1, prev2[i1]);
                    let out3 = prev[i3].mul_add(consts::VERT_MUL_PREV_3, prev2[i3]);
                    let out5 = prev[i5].mul_add(consts::VERT_MUL_PREV_5, prev2[i5]);

                    let out1 = sum.mul_add(consts::VERT_MUL_IN_1, -out1);
                    let out3 = sum.mul_add(consts::VERT_MUL_IN_3, -out3);
                    let out5 = sum.mul_add(consts::VERT_MUL_IN_5, -out5);

                    out[i1] = out1;
                    out[i3] = out3;
                    out[i5] = out5;

                    if n >= 0 {
                        output[n as usize * width + i] = out1 + out3 + out5;
                    }
                }

                prev2.copy_from_slice(&prev);
                prev.copy_from_slice(&out);

                n += 1;
            }
        }
    }
}
