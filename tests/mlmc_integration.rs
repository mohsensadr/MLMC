use mlmc_sde::{mlmc::mlmc, sde::SDEparams};

#[test]
fn test_full_pipeline() {
    let params = SDEparams { p: 1.0, t: 0.1, sig: 1.0, x0: 1.2 };
    let (P, Nl, _, _, _, _, _) = mlmc(3, 5, 50, 1e-2, &params);

    assert!(Nl.len() >= 3);
    assert!(P.is_finite());
}