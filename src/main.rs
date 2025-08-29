use ndarray::Array1;
use ndarray_rand::rand_distr::{Normal, Distribution};
use ndarray_rand::rand::thread_rng;

fn sde(l: usize, n: usize, p: f64, t: f64, sig: f64, x0: f64) -> [f64; 7] {
    let nf = 2_usize.pow((l - 1) as u32);
    let hf = t / nf as f64;

    let mut sums = [0.0; 7];
    let mut xf = Array1::<f64>::from_elem(n, x0);
    let mut xc = Array1::<f64>::from_elem(n, x0);

    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = thread_rng();

    // Helper function to generate N normal samples scaled by sqrt(hf)
    let mut generate_normal = |n: usize| -> Array1<f64> {
        Array1::from_iter((0..n).map(|_| normal.sample(&mut rng) * hf.sqrt()))
    };

    if l == 1 {
        let dwf = generate_normal(n);
        for i in 0..n {
            xf[i] = xf[i] - xf[i].powf(p) * hf + (2.0f64).sqrt() * sig * dwf[i];
        }
    } else {
        let nc = nf / 2;
        let hc = t / nc as f64;
        for _ in 0..nc {
            let dwf0 = generate_normal(n);
            let dwf1 = generate_normal(n);
            for i in 0..n {
                xf[i] = xf[i] - xf[i].powf(p) * hf + (2.0f64).sqrt() * sig * dwf0[i];
                xf[i] = xf[i] - xf[i].powf(p) * hf + (2.0f64).sqrt() * sig * dwf1[i];
                let dwc = dwf0[i] + dwf1[i];
                xc[i] = xc[i] - xc[i].powf(p) * hc + (2.0f64).sqrt() * sig * dwc;
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


use linregress::{FormulaRegressionBuilder, RegressionDataBuilder};
fn mlmc(
    Lmin: usize,
    Lmax: usize,
    N0: usize,
    eps: f64,
    p: f64,
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
                let sums = sde(l + 1, dNl[l], p, 0.1, 1.0, 1.2);
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
            dNl[l] = ((Vl[l] / Cl[l]).sqrt() * s / ((1.0 - theta) * eps * eps) - suml[0][l])
                .max(0.0)
                .ceil() as usize;
        }

        // Linear regression for alpha, beta, gamma
        if L > 1 {
            let xl: Vec<f64> = (2..=L).map(|l| l as f64).collect();
            let yl_alpha: Vec<f64> = (1..L).map(|l| -ml[l].log2()).collect();
            let columns_alpha: Vec<(&str, Vec<f64>)> = vec![
                ("x", xl.clone()),
                ("y", yl_alpha.clone()),
            ];

            let data_alpha = RegressionDataBuilder::new()
                .build_from(columns_alpha)
                .unwrap();

            // Fit the regression
            let formula_alpha = FormulaRegressionBuilder::new()
                .data(&data_alpha)
                .formula("y ~ x")
                .fit()
                .unwrap();

            let idx = formula_alpha.parameters.regressor_names
                .iter()
                .position(|name| name == "x")
                .expect("x not found in regression");
            // Extract slope as alpha
            alpha = formula_alpha.parameters.regressor_values[idx].max(0.5);

            // Beta regression
            let yl_beta: Vec<f64> = (1..L).map(|l| -Vl[l].log2()).collect();
            let columns_beta: Vec<(&str, Vec<f64>)> = vec![
                ("x", xl.clone()),
                ("y", yl_beta.clone()),
            ];

            let data_beta = RegressionDataBuilder::new()
                .build_from(columns_beta)
                .unwrap();

            let formula_beta = FormulaRegressionBuilder::new()
                .data(&data_beta)
                .formula("y ~ x")
                .fit()
                .unwrap();

            let idx_beta = formula_beta.parameters.regressor_names
                .iter()
                .position(|name| name == "x")
                .expect("x not found in regression");

            beta = formula_beta.parameters.regressor_values[idx_beta].max(0.5);

            // Gamma regression
            let yl_gamma: Vec<f64> = (1..L).map(|l| Cl[l].log2()).collect();
            let columns_gamma: Vec<(&str, Vec<f64>)> = vec![
                ("x", xl.clone()),
                ("y", yl_gamma.clone()),
            ];

            let data_gamma = RegressionDataBuilder::new()
                .build_from(columns_gamma)
                .unwrap();

            let formula_gamma = FormulaRegressionBuilder::new()
                .data(&data_gamma)
                .formula("y ~ x")
                .fit()
                .unwrap();

            let idx_gamma = formula_gamma.parameters.regressor_names
                .iter()
                .position(|name| name == "x")
                .expect("x not found in regression");

            gamma = formula_gamma.parameters.regressor_values[idx_gamma].max(0.5);
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
                    Vl.push(Vl[L - 2] / 2.0f64.powf(beta));
                    Cl.push(Cl[L - 2] * 2.0f64.powf(gamma));
                    NlCl.push(0.0);
                    suml[0].push(0.0);
                    suml[1].push(0.0);
                    suml[2].push(0.0);
                    dNl.push(((Vl[L - 1] / Cl[L - 1]).sqrt() * s / ((1.0 - theta) * eps * eps)).ceil() as usize);
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


fn main() {
    //let sum = sde(3, 2000, 1.0, 0.1, 1.0, 1.2);
    //print!("{:#?}", sum);
    
    let p = 1.0;
    let eps_list = vec![1e-2];
    let mut levels = vec![0_usize; eps_list.len()];
    let mut costs = vec![0.0; eps_list.len()];
    let mut costs_sd = vec![0.0; eps_list.len()];

    for (i, &eps) in eps_list.iter().enumerate() {
        let (P, Nl, Cl, Vl, alpha, beta, gamma) = mlmc(3, 10, 100, eps, p);
        let L = Nl.len();
        levels[i] = L;
        costs[i] = (0..L).map(|l| Nl[l] as f64 * Cl[l]).sum();

        // Standard MC at finest level
        let Ntest = 10000;
        let sums = sde(L, Ntest, p, 0.1, 1.0, 1.2);
        let varL = ((sums[6] / Ntest as f64) - (sums[5] / Ntest as f64).powi(2)).max(1e-10);
        let NlCl_test = sums[0];
        costs_sd[i] = NlCl_test * varL / (eps * eps);
    }

    println!("Levels per eps: {:?}", levels);
    println!("MLMC costs: {:?}", costs);
    println!("MC costs: {:?}", costs_sd);
    
}
