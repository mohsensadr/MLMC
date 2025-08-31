/// Performs a Multilevel Monte Carlo (MLMC) simulation to estimate the expected value of an SDE.
///
/// # Parameters
/// - `lmin`: Minimum level of discretization (coarsest level). Must be ≥ 3.
/// - `lmax`: Maximum level of discretization (finest level). Must be ≥ `lmin`.
/// - `n0`: Initial number of Monte Carlo samples per level. Must be > 0.
/// - `eps`: Desired root-mean-square accuracy for the MLMC estimate. Must be > 0.
/// - `sde_params`: Reference to `SDEparams` struct containing the SDE parameters.
///
/// # Returns
/// A tuple containing:
/// 1. `p`: Estimated expected value of the SDE at time `t`.
/// 2. `nl`: Vector of number of samples used at each level (as f64).
/// 3. `cl`: Vector of computational costs per sample at each level.
/// 4. `vl`: Vector of variances of the multilevel differences at each level.
/// 5. `alpha`: Estimated weak convergence rate (from regression).
/// 6. `beta`: Estimated variance convergence rate (from regression).
/// 7. `gamma`: Estimated cost growth rate per level (from regression).
///
/// # Behavior
/// - Runs a multilevel Monte Carlo estimator using the Euler-Maruyama discretization.
/// - Automatically adapts the number of samples per level based on variance and cost to achieve the target accuracy `eps`.
/// - Estimates convergence rates `alpha`, `beta`, and `gamma` using linear regression across levels.
/// - Expands to finer levels if weak convergence is not achieved at the current finest level.
///
/// # Panics
/// Panics if input parameters are invalid (e.g., `lmin < 3`, `lmax < lmin`, `n0 == 0`, or `eps <= 0`).
///
/// # Example
/// ```
/// use mlmc_sde::{mlmc::mlmc, sde::SDEparams};
///
/// let params = SDEparams { p: 1.0, t: 0.1, sig: 1.0, x0: 1.2 };
/// let (p, nl, cl, vl, alpha, beta, gamma) = mlmc(3, 5, 100, 1e-2, &params);
///
/// println!("MLMC estimate: {}", p);
/// println!("Number of samples per level: {:?}", nl);
/// println!("Estimated alpha: {}, beta: {}, gamma: {}", alpha, beta, gamma);
/// ```

use crate::sde::{sde, SDEparams};
use crate::utility::linear_regression;

pub fn mlmc(
    lmin: usize,
    lmax: usize,
    n0: usize,
    eps: f64,
    sde_params: &SDEparams,
) -> (f64, Vec<f64>, Vec<f64>, Vec<f64>, f64, f64, f64) {
    let mut suml = vec![vec![0.0; lmax]; 3];
    let mut dnl = vec![0_usize; lmax];
    let mut ml = vec![0.0; lmax];
    let mut vl = vec![0.0; lmax];
    let mut nlcl = vec![0.0; lmax];

    let mut nl = vec![0_usize; lmax];
    let mut cl = vec![0.0; lmax];

    if lmin < 3 || lmax < lmin || n0 == 0 || eps <= 0.0 {
        panic!("Invalid input parameters");
    }

    let mut curr_l_max = lmin;
    let mut alpha = 0.0;
    let mut beta = 0.0;
    let mut gamma = 0.0;
    let theta = 0.25;

    for l in 0..lmin {
        dnl[l] = n0;
    }

    let mut converged = false;
    while !converged {
        for l in 0..curr_l_max {
            if dnl[l] > 0 {
                let sums = sde(l + 1, dnl[l], &sde_params);
                suml[0][l] += dnl[l] as f64;
                suml[1][l] += sums[1];
                suml[2][l] += sums[2];
                nlcl[l] += sums[0];
            }
        }

        for l in 0..curr_l_max {
            if suml[0][l] > 0.0 {
                ml[l] = (suml[1][l] / suml[0][l]).abs();
                vl[l] = (suml[2][l] / suml[0][l] - ml[l].powi(2)).max(0.0);
                cl[l] = nlcl[l] / suml[0][l];
            }
            if l > 1 {
                ml[l] = ml[l].max(0.5 * ml[l - 1] / 2.0f64.powf(alpha));
                vl[l] = vl[l].max(0.5 * vl[l - 1] / 2.0f64.powf(beta));
            }
        }

        let s: f64 = (0..curr_l_max).map(|l| (vl[l] * cl[l]).sqrt()).sum();
        for l in 0..curr_l_max {
            dnl[l] = (((vl[l] / cl[l]).sqrt() * s / ((1.0 - theta) * eps * eps) - suml[0][l]) as usize)
                .max(0);
        }

        if curr_l_max > 1 {
            let xl: Vec<f64> = (1..curr_l_max).map(|l| l as f64).collect();
            let yl_alpha: Vec<f64> = (1..curr_l_max).map(|l| -ml[l].log2()).collect();
            if let Ok((_, slope)) = linear_regression(&xl, &yl_alpha) {
                alpha = slope.max(0.5);
            }

            let yl_beta: Vec<f64> = (1..curr_l_max).map(|l| -vl[l].log2()).collect();
            if let Ok((_, slope)) = linear_regression(&xl, &yl_beta) {
                beta = slope.max(0.5);
            }

            let yl_gamma: Vec<f64> = (1..curr_l_max).map(|l| cl[l].log2()).collect();
            if let Ok((_, slope)) = linear_regression(&xl, &yl_gamma) {
                gamma = slope.max(0.5);
            }
        }

        let sr: f64 = (0..curr_l_max)
            .map(|l| (dnl[l] as f64 - 0.01 * suml[0][l]).max(0.0))
            .sum();

        if sr.abs() < 1e-5 {
            converged = true;
            let rem = ml[curr_l_max - 1] / (2.0f64.powf(alpha) - 1.0);
            if rem > (theta.sqrt() * eps) {
                if curr_l_max == lmax {
                    println!("*** failed to achieve weak convergence ***");
                } else {
                    converged = false;
                    curr_l_max += 1;
                    vl[curr_l_max - 1] = vl[curr_l_max - 2] / 2.0f64.powf(beta);
                    cl[curr_l_max - 1] = cl[curr_l_max - 2] * 2.0f64.powf(gamma);
                    for l in 0..curr_l_max {
                        dnl[l] = (((vl[l] / cl[l]).sqrt() * s / ((1.0 - theta) * eps * eps) - suml[0][l]) as usize)
                            .max(0);
                    }
                }
            }
        }
    }

    let mut p = 0.0;
    for l in 0..curr_l_max {
        p += suml[1][l] / suml[0][l];
        nl[l] = suml[0][l] as usize;
        cl[l] = nlcl[l] / suml[0][l];
    }

    (
        p,
        nl[..curr_l_max].iter().map(|&x| x as f64).collect(),
        cl[..curr_l_max].to_vec(),
        vl[..curr_l_max].to_vec(),
        alpha,
        beta,
        gamma,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sde::SDEparams;

    #[test]
    fn test_mlmc_runs() {
        let params = SDEparams { p: 1.0, t: 0.1, sig: 1.0, x0: 1.2 };
        let (P, nl, cl, vl, alpha, beta, gamma) = mlmc(3, 5, 10, 1e-2, &params);

        assert!(nl.len() >= 3);       // Should use at least Lmin levels
        assert!(P.is_finite());       // Expect finite estimate
        assert!(alpha >= 0.5);        // Constraints from regression
        assert!(beta >= 0.5);
        assert!(gamma >= 0.5);
    }
}