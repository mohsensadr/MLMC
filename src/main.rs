//! Main entry point for the MLMC simulation.
//!
//! This program runs a Multilevel Monte Carlo (MLMC) simulation for a simple SDE
//! and compares it with standard Monte Carlo (MC) at the finest level. The results
//! are written to two CSV files for further analysis.
//!
//! # Behavior
//! 1. Defines SDE parameters (`SDEparams`) and MLMC simulation parameters (`lmin`, `lmax`, `n0`, etc.).
//! 2. Iterates over a list of target accuracies (`eps_list`) and runs the MLMC estimator for each `eps`.
//! 3. Computes MLMC estimates (`P`), sample counts per level (`Nl`), costs (`Cl`), and variances (`Vl`).
//! 4. Performs a standard MC simulation at the finest level to estimate the cost for the same accuracy.
//! 5. Writes summary results to `mlmc_vs_mc.csv` and detailed level information to `mlmc_details.csv`.
//!
//! # Output Files
//! - `mlmc_vs_mc.csv`: Columns `eps,levels,mlmc_cost,mc_cost` for comparison of MLMC vs MC.
//! - `mlmc_details.csv`: Columns `P,eps,alpha,beta,gamma,L,Nl` with detailed per-level data.
//!
//! # Example
//! ```no_run
//! // Run the program:
//! cargo run
//! // This will generate the CSV files with MLMC results.
//! ```

mod sde;
mod utility;
mod mlmc;

use sde::SDEparams;
use mlmc::mlmc;

use std::fs::File;
use std::io::{Write, BufWriter};

fn main() -> std::io::Result<()> {
    let file = File::create("mlmc_vs_mc.csv")?;
    let file2 = File::create("mlmc_details.csv")?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "eps,levels,mlmc_cost,mc_cost")?;
    let mut writer2 = BufWriter::new(file2);
    writeln!(writer2, "P,eps,alpha,beta,gamma,L,Nl")?;

    let lmin: usize = 3;
    let lmax: usize = 10;
    let n0: usize = 100;
    let sde_params = SDEparams {
        p: 1.0,
        t: 0.1,
        sig: 1.0,
        x0: 1.2,
    };

    let eps_list = vec![1e-2, 1e-3, 1e-4];
    let mut levels = vec![0_usize; eps_list.len()];
    let mut costs = vec![0.0; eps_list.len()];
    let mut costs_sd = vec![0.0; eps_list.len()];

    for (i, &eps) in eps_list.iter().enumerate() {
        println!("-----------------\n\nRunning MLMC for eps = {}\n", eps);

        let (p, nl, cl, vl, alpha, beta, gamma) = mlmc(lmin, lmax, n0, eps, &sde_params);
        let final_l = nl.len();
        levels[i] = final_l;
        costs[i] = (0..final_l).map(|l| nl[l] as f64 * cl[l]).sum();

        println!("P: {:?}", p);
        println!("Number of samples per level: {:#?}", nl);
        println!("Cost per sample per level: {:#?}", cl);
        println!("Variance for each sample per level: {:#?}", vl);
        println!("alpha: {:?}", alpha);
        println!("beta: {:?}", beta);
        println!("gamma: {:?}", gamma);

        // Standard MC at finest level
        let ntest = 10000;
        let sums = crate::sde::sde(final_l, ntest, &sde_params);
        let varl = ((sums[6] / ntest as f64) - (sums[5] / ntest as f64).powi(2)).max(1e-10);
        let nlcl_test = sums[0];
        costs_sd[i] = nlcl_test * varl / (eps * eps);

        writeln!(writer, "{},{},{},{}", eps, levels[i], costs[i], costs_sd[i])?;
        writeln!(
            writer2,
            "{},{},{},{},{},{},{}",
            p,
            eps,
            alpha,
            beta,
            gamma,
            final_l,
            nl.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ")
        )?;
    }

    Ok(())
}
