/// Simulates a stochastic differential equation (SDE) using a multilevel discretization.
///
/// # Parameters
/// - `l`: The level of discretization (l=1 is the coarsest). Higher levels refine the time step.
/// - `n`: Number of Monte Carlo samples to simulate.
/// - `sde_params`: Reference to `SDEparams` struct containing:
///     - `p`: Power in the drift term `-x^p`.
///     - `t`: Total time of simulation.
///     - `sig`: Noise intensity (standard deviation) of the Brownian term.
///     - `x0`: Initial condition of the SDE.
///
/// # Returns
/// An array of 7 `f64` values summarizing the simulation:
/// 1. `nf * n` — total number of fine steps multiplied by number of samples.
/// 2. Sum of differences `dp` (difference between fine and coarse paths for l>1, or fine path for l=1).
/// 3. Sum of squares of differences `dp^2`.
/// 4. Sum of cubes of differences `dp^3`.
/// 5. Sum of fourth powers of differences `dp^4`.
/// 6. Sum of final fine path values `xf`.
/// 7. Sum of squares of final fine path values `xf^2`.
///
/// # Behavior
/// - For `l == 1`, only a single coarse path is simulated.
/// - For `l > 1`, both fine and coarse paths are simulated to compute multilevel differences.
/// - The SDE is discretized using an Euler-Maruyama scheme with Brownian increments scaled by the time step.
///
/// # Example
/// ```
/// use mlmc_sde::{sde, SDEparams};
///
/// let params = SDEparams { p: 1.0, t: 0.1, sig: 1.0, x0: 1.2 };
/// let result = sde(2, 1000, &params);
/// assert_eq!(result.len(), 7);
/// ```
use ndarray::Array1;
use ndarray_rand::rand_distr::{Normal, Distribution};
use ndarray_rand::rand::thread_rng;

pub struct SDEparams {
    pub p: f64,
    pub t: f64,
    pub sig: f64,
    pub x0: f64,
}

pub fn sde(l: usize, n: usize, sde_params: &SDEparams) -> [f64; 7] {
    let nf = 2_usize.pow((l - 1) as u32);
    let hf = sde_params.t / nf as f64;

    let mut sums = [0.0; 7];
    let mut xf = Array1::<f64>::from_elem(n, sde_params.x0);
    let mut xc = Array1::<f64>::from_elem(n, sde_params.x0);

    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = thread_rng();

    let mut generate_normal = |n: usize| -> Array1<f64> {
        Array1::from_iter((0..n).map(|_| normal.sample(&mut rng) * hf.sqrt()))
    };

    if l == 1 {
        let dwf = generate_normal(n);
        for i in 0..n {
            xf[i] = xf[i] - xf[i].powf(sde_params.p) * hf + (2.0f64).sqrt() * sde_params.sig * dwf[i];
        }
    } else {
        let nc = nf / 2;
        let hc = sde_params.t / nc as f64;
        for _ in 0..nc {
            let dwf0 = generate_normal(n);
            let dwf1 = generate_normal(n);
            for i in 0..n {
                xf[i] = xf[i] - xf[i].powf(sde_params.p) * hf + (2.0f64).sqrt() * sde_params.sig * dwf0[i];
                xf[i] = xf[i] - xf[i].powf(sde_params.p) * hf + (2.0f64).sqrt() * sde_params.sig * dwf1[i];
                let dwc = dwf0[i] + dwf1[i];
                xc[i] = xc[i] - xc[i].powf(sde_params.p) * hc + (2.0f64).sqrt() * sde_params.sig * dwc;
            }
        }
    }

    let mut dp = Array1::<f64>::zeros(n);
    for i in 0..n {
        dp[i] = if l == 1 { xf[i] } else { xf[i] - xc[i] };
    }

    sums[0] = nf as f64 * n as f64;
    sums[1] = dp.sum();
    sums[2] = dp.mapv(|x| x.powi(2)).sum();
    sums[3] = dp.mapv(|x| x.powi(3)).sum();
    sums[4] = dp.mapv(|x| x.powi(4)).sum();
    sums[5] = xf.sum();
    sums[6] = xf.mapv(|x| x.powi(2)).sum();

    sums
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sde_level1_runs() {
        let params = SDEparams { p: 1.0, t: 0.1, sig: 1.0, x0: 1.2 };
        let result = sde(1, 100, &params);
        assert_eq!(result.len(), 7);
    }

    #[test]
    fn test_sde_higher_level_runs() {
        let params = SDEparams { p: 1.0, t: 0.1, sig: 1.0, x0: 1.2 };
        let result = sde(3, 50, &params);
        assert_eq!(result.len(), 7);
    }
}