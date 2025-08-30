use ndarray::Array1;
use ndarray_rand::rand_distr::{Normal, Distribution};
use ndarray_rand::rand::thread_rng;

struct SDEparams {
    p: f64,
    t: f64,
    sig: f64,
    x0: f64,
}

fn sde(l: usize, n: usize, sde_params: &SDEparams) -> [f64; 7] {
    let nf = 2_usize.pow((l - 1) as u32);
    let hf = sde_params.t / nf as f64;

    let mut sums = [0.0; 7];
    let mut xf = Array1::<f64>::from_elem(n, sde_params.x0);
    let mut xc = Array1::<f64>::from_elem(n, sde_params.x0);

    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = thread_rng();

    // Helper function to generate N normal samples scaled by sqrt(hf)
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


/// Fit a simple linear regression y = intercept + slope * x
/// Returns (intercept, slope)
fn linear_regression(x: &[f64], y: &[f64]) -> Result<(f64, f64), &'static str> {
    if x.len() != y.len() {
        return Err("x and y must have the same length");
    }
    let n = x.len();
    if n < 2 {
        return Err("Need at least two points for regression");
    }

    // Special case: exactly two points → fit line directly
    if n == 2 {
        let slope = (y[1] - y[0]) / (x[1] - x[0]);
        let intercept = y[0] - slope * x[0];
        return Ok((intercept, slope));
    }

    // General case: OLS formulas
    let mean_x = x.iter().sum::<f64>() / n as f64;
    let mean_y = y.iter().sum::<f64>() / n as f64;

    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..n {
        let dx = x[i] - mean_x;
        num += dx * (y[i] - mean_y);
        den += dx * dx;
    }

    if den == 0.0 {
        return Err("All x values are identical; cannot fit line");
    }

    let slope = num / den;
    let intercept = mean_y - slope * mean_x;

    Ok((intercept, slope))
}


fn mlmc(
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

            match linear_regression(&xl, &yl_alpha) {
                Ok((intercept, slope)) => {
                    alpha = slope.max(0.5);
                }
                Err(e) => {
                    println!("Error: {}", e);
                }
            }

            // Beta regression
            let yl_beta: Vec<f64> = (1..L).map(|l| -Vl[l].log2()).collect();
            match linear_regression(&xl, &yl_beta) {
                Ok((intercept, slope)) => {
                    beta = slope.max(0.5);
                }
                Err(e) => {
                    println!("Error: {}", e);
                }
            }

            // Gamma regression
            let yl_gamma: Vec<f64> = (1..L).map(|l| Cl[l].log2()).collect();
            match linear_regression(&xl, &yl_gamma) {
                Ok((intercept, slope)) => {
                    gamma = slope.max(0.5);
                }
                Err(e) => {
                    println!("Error: {}", e);
                }
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

use std::fs::File;
use std::io::{Write, BufWriter};



fn main() -> std::io::Result<()>  {    
    let file = File::create("mlmc_vs_mc.csv")?;
    let file2 = File::create("mlmc_details.csv")?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "eps,levels,mlmc_cost,mc_cost")?;
    let mut writer2 = BufWriter::new(file2);
    writeln!(writer2, "P,eps,alpha,beta,gamma,L,Nl")?;

    let Lmin: usize = 3;
    let Lmax: usize =10;
    let N0: usize = 100;
    let sde_params = SDEparams {
        p: 1.0,
        t: 0.1,
        sig: 1.0,
        x0: 1.2,
    };

    let eps_list = vec![1e-2, 1e-3];
    let mut levels = vec![0_usize; eps_list.len()];
    let mut costs = vec![0.0; eps_list.len()];
    let mut costs_sd = vec![0.0; eps_list.len()];

    for (i, &eps) in eps_list.iter().enumerate() {
        print!("Running MLMC for eps = {}\n", eps);

        let (P, Nl, Cl, Vl, alpha, beta, gamma) = mlmc(Lmin, Lmax, N0, eps, &sde_params);
        let L = Nl.len();
        levels[i] = L;
        costs[i] = (0..L).map(|l| Nl[l] as f64 * Cl[l]).sum();

        println!("P: {:?}", P);
        println!("P: {:#?}", Nl);
        println!("alpha: {:?}", alpha);
        println!("beta: {:?}", beta);
        println!("gamma: {:?}", gamma);

        // Standard MC at finest level
        let Ntest = 10000;
        let sums = sde(L, Ntest, &sde_params);
        let varL = ((sums[6] / Ntest as f64) - (sums[5] / Ntest as f64).powi(2)).max(1e-10);
        let NlCl_test = sums[0];
        costs_sd[i] = NlCl_test * varL / (eps * eps);

        // write row into CSV
        writeln!(writer, "{},{},{},{}", eps, levels[i], costs[i], costs_sd[i])?;
        writeln!(writer2, "{},{},{},{},{},{},{}", P, eps, alpha, beta, gamma, L, Nl.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","))?;
    }

    Ok(())
}
