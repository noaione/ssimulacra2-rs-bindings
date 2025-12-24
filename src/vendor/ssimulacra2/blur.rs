use gaussian::RecursiveGaussian;

/// Structure handling image blur.
///
/// This struct contains the necessary buffers and the kernel used for blurring
/// (currently a recursive approximation of the Gaussian filter).
///
/// Note that the width and height of the image passed to [blur][Self::blur] needs to exactly
/// match the width and height of this instance. If you reduce the image size (e.g. via
/// downscaling), [`shrink_to`][Self::shrink_to] can be used to resize the internal buffers.
pub struct Blur {
    kernel: RecursiveGaussian,
    temp: Vec<f32>,
    width: usize,
    height: usize,
}

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

mod gaussian {
    mod consts {
        #![allow(clippy::unreadable_literal)]
        include!(concat!(env!("OUT_DIR"), "/recursive_gaussian.rs"));
    }

    /// Implements "Recursive Implementation of the Gaussian Filter Using Truncated
    /// Cosine Functions" by Charalampidis [2016].
    pub struct RecursiveGaussian;

    impl RecursiveGaussian {
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
                self.vertical_pass::<J>(&input[x..], &mut output[x..], width, height);
                x += J;
            }

            while x + K <= width {
                self.vertical_pass::<K>(&input[x..], &mut output[x..], width, height);
                x += K;
            }

            while x < width {
                self.vertical_pass::<1>(&input[x..], &mut output[x..], width, height);
                x += 1;
            }
        }

        // Apply 1D vertical scan on COLUMNS elements at a time
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
