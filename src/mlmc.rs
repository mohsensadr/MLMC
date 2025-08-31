use crate::sde::{sde, SDEparams};
use crate::utility::linear_regression;

pub fn mlmc(
    Lmin: usize,
    Lmax: usize,
    N0: usize,
    eps: f64,
    sde_params: &SDEparams,
) -> (f64, Vec<f64>, Vec<f64>, Vec<f64>, f64, f64, f64) {
    let mut suml = vec![vec![0.0; Lmax]; 3];
    let mut dNl = vec![0_usize; Lmax];
    let mut ml = vec![0.0; Lmax];
    let mut Vl = vec![0.0; Lmax];
    let mut NlCl = vec![0.0; Lmax];

    let mut Nl = vec![0_usize; Lmax];
    let mut Cl = vec![0.0; Lmax];

    if Lmin < 3 || Lmax < Lmin || N0 == 0 || eps <= 0.0 {
        panic!("Invalid input parameters");
    }

    let mut L = Lmin;
    let mut alpha = 0.0;
    let mut beta = 0.0;
    let mut gamma = 0.0;
    let theta = 0.25;

    for l in 0..Lmin {
        dNl[l] = N0;
    }

    let mut converged = false;
    while !converged {
        for l in 0..L {
            if dNl[l] > 0 {
                let sums = sde(l + 1, dNl[l], &sde_params);
                suml[0][l] += dNl[l] as f64;
                suml[1][l] += sums[1];
                suml[2][l] += sums[2];
                NlCl[l] += sums[0];
            }
        }

        for l in 0..L {
            if suml[0][l] > 0.0 {
                ml[l] = (suml[1][l] / suml[0][l]).abs();
                Vl[l] = (suml[2][l] / suml[0][l] - ml[l].powi(2)).max(0.0);
                Cl[l] = NlCl[l] / suml[0][l];
            }
            if l > 1 {
                ml[l] = ml[l].max(0.5 * ml[l - 1] / 2.0f64.powf(alpha));
                Vl[l] = Vl[l].max(0.5 * Vl[l - 1] / 2.0f64.powf(beta));
            }
        }

        let s: f64 = (0..L).map(|l| (Vl[l] * Cl[l]).sqrt()).sum();
        for l in 0..L {
            dNl[l] = (((Vl[l] / Cl[l]).sqrt() * s / ((1.0 - theta) * eps * eps) - suml[0][l]) as usize)
                .max(0);
        }

        if L > 1 {
            let xl: Vec<f64> = (1..L).map(|l| l as f64).collect();
            let yl_alpha: Vec<f64> = (1..L).map(|l| -ml[l].log2()).collect();
            if let Ok((_, slope)) = linear_regression(&xl, &yl_alpha) {
                alpha = slope.max(0.5);
            }

            let yl_beta: Vec<f64> = (1..L).map(|l| -Vl[l].log2()).collect();
            if let Ok((_, slope)) = linear_regression(&xl, &yl_beta) {
                beta = slope.max(0.5);
            }

            let yl_gamma: Vec<f64> = (1..L).map(|l| Cl[l].log2()).collect();
            if let Ok((_, slope)) = linear_regression(&xl, &yl_gamma) {
                gamma = slope.max(0.5);
            }
        }

        let sr: f64 = (0..L)
            .map(|l| (dNl[l] as f64 - 0.01 * suml[0][l]).max(0.0))
            .sum();

        if sr.abs() < 1e-5 {
            converged = true;
            let rem = ml[L - 1] / (2.0f64.powf(alpha) - 1.0);
            if rem > (theta.sqrt() * eps) {
                if L == Lmax {
                    println!("*** failed to achieve weak convergence ***");
                } else {
                    converged = false;
                    L += 1;
                    Vl[L - 1] = Vl[L - 2] / 2.0f64.powf(beta);
                    Cl[L - 1] = Cl[L - 2] * 2.0f64.powf(gamma);
                    for l in 0..L {
                        dNl[l] = (((Vl[l] / Cl[l]).sqrt() * s / ((1.0 - theta) * eps * eps) - suml[0][l]) as usize)
                            .max(0);
                    }
                }
            }
        }
    }

    let mut P = 0.0;
    for l in 0..L {
        P += suml[1][l] / suml[0][l];
        Nl[l] = suml[0][l] as usize;
        Cl[l] = NlCl[l] / suml[0][l];
    }

    (
        P,
        Nl[..L].iter().map(|&x| x as f64).collect(),
        Cl[..L].to_vec(),
        Vl[..L].to_vec(),
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
        let (P, Nl, Cl, Vl, alpha, beta, gamma) = mlmc(3, 5, 10, 1e-2, &params);

        assert!(Nl.len() >= 3);       // Should use at least Lmin levels
        assert!(P.is_finite());       // Expect finite estimate
        assert!(alpha >= 0.5);        // Constraints from regression
        assert!(beta >= 0.5);
        assert!(gamma >= 0.5);
    }
}