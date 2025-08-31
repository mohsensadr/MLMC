#![allow(non_snake_case)]
#![allow(unused_variables)]

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

    let Lmin: usize = 3;
    let Lmax: usize = 10;
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
        println!("-----------------\n\nRunning MLMC for eps = {}\n", eps);

        let (P, Nl, Cl, Vl, alpha, beta, gamma) = mlmc(Lmin, Lmax, N0, eps, &sde_params);
        let L = Nl.len();
        levels[i] = L;
        costs[i] = (0..L).map(|l| Nl[l] as f64 * Cl[l]).sum();

        println!("P: {:?}", P);
        println!("Nl: {:#?}", Nl);
        println!("alpha: {:?}", alpha);
        println!("beta: {:?}", beta);
        println!("gamma: {:?}", gamma);

        // Standard MC at finest level
        let Ntest = 10000;
        let sums = crate::sde::sde(L, Ntest, &sde_params);
        let varL = ((sums[6] / Ntest as f64) - (sums[5] / Ntest as f64).powi(2)).max(1e-10);
        let NlCl_test = sums[0];
        costs_sd[i] = NlCl_test * varL / (eps * eps);

        writeln!(writer, "{},{},{},{}", eps, levels[i], costs[i], costs_sd[i])?;
        writeln!(
            writer2,
            "{},{},{},{},{},{},{}",
            P,
            eps,
            alpha,
            beta,
            gamma,
            L,
            Nl.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ")
        )?;
    }

    Ok(())
}
