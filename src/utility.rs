/// Performs a simple linear regression to fit a line `y = intercept + slope * x`.
///
/// # Parameters
/// - `x`: Slice of independent variable values.
/// - `y`: Slice of dependent variable values. Must have the same length as `x`.
///
/// # Returns
/// - `Ok((intercept, slope))` containing the estimated line parameters.
/// - `Err(&'static str)` if the input is invalid:
///     - `x` and `y` have different lengths.
///     - Less than two points are provided.
///     - All `x` values are identical (cannot fit a line).
///
/// # Behavior
/// - If exactly two points are provided, computes slope and intercept directly.
/// - If more than two points, uses ordinary least squares formulas:
///   - Slope: `sum((x - mean_x)*(y - mean_y)) / sum((x - mean_x)^2)`
///   - Intercept: `mean_y - slope * mean_x`
///
/// # Example
/// ```
/// use mlmc_sde::utility::linear_regression;
///
/// let x = vec![1.0, 2.0, 3.0];
/// let y = vec![2.0, 4.0, 6.0];
/// let (intercept, slope) = linear_regression(&x, &y).unwrap();
/// assert!((slope - 2.0).abs() < 1e-10);
/// assert!((intercept - 0.0).abs() < 1e-10);
/// ```
///
pub fn linear_regression(x: &[f64], y: &[f64]) -> Result<(f64, f64), &'static str> {
    if x.len() != y.len() {
        return Err("x and y must have the same length");
    }
    let n = x.len();
    if n < 2 {
        return Err("Need at least two points for regression");
    }

    if n == 2 {
        let slope = (y[1] - y[0]) / (x[1] - x[0]);
        let intercept = y[0] - slope * x[0];
        return Ok((intercept, slope));
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_point_regression() {
        let x = vec![1.0, 2.0];
        let y = vec![2.0, 4.0];
        let (intercept, slope) = linear_regression(&x, &y).unwrap();
        assert!((slope - 2.0).abs() < 1e-10);
        assert!((intercept - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_three_point_regression() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![2.0, 4.0, 6.0];
        let (intercept, slope) = linear_regression(&x, &y).unwrap();
        assert!((slope - 2.0).abs() < 1e-10);
        assert!((intercept - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_invalid_regression() {
        let x = vec![1.0];
        let y = vec![2.0];
        assert!(linear_regression(&x, &y).is_err());
    }
}