pub enum Filter {
    Gaussian(f64),

}


impl Filter {
    pub fn apply(&self, input: &[Vec<f64>]) -> Vec<Vec<f64>> {
        match self {
            Filter::Gaussian(sigma) => Self::gaussian_filter(input, *sigma),
        }
    }

    fn gaussian_filter(input: &[Vec<f64>], sigma: f64) -> Vec<Vec<f64>> {
        let kernel_size = 5;
        let mut kernel = vec![vec![0.0; kernel_size]; kernel_size];
        let mut sum = 0.0;

        let center = kernel_size as isize / 2;
        let sigma2 = 2.0 * sigma * sigma;

        for i in 0..kernel_size {
            for j in 0..kernel_size {
                let x = (i as isize - center) as f64;
                let y = (j as isize - center) as f64;

                kernel[i][j] = (- (x*y + y*y) / sigma2).exp();
                sum += kernel[i][j];
            }
        }

        for i in 0..kernel_size {
            for j in 0..kernel_size {
                kernel[i][j] /=sum;

            }
        }

        return Self::convolve(input, &kernel)
    }


    fn convolve(input: &[Vec<f64>], kernel: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let rows = input.len();
        let cols = input[0].len();
        
        let k_size = kernel.len();
        let pad = k_size /2;

        let mut output = vec![vec![0.0; cols]; rows];

        for i in pad..rows - pad {
            for j in pad..cols - pad {
                let mut sum = 0.0;
                for ki in 0..k_size {
                    for kj in 0..k_size {
                        let x = i + ki - pad;
                        let y = j + kj - pad;
                        sum += input[x][y] * kernel[ki][kj];
                    }
                }

                output[i][j] = sum;
            }
        }

        return output
    }
}