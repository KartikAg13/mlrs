use ndarray::Array1;

pub fn r2_score(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let mean = y_true.mean().unwrap_or(0.0);
    let ss_tot: f64 = y_true.iter().map(|&v| (v - mean).powi(2)).sum();
    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| (t - p).powi(2))
        .sum();
    if ss_tot < f64::EPSILON {
        1.0
    } else {
        1.0 - ss_res / ss_tot
    }
}
