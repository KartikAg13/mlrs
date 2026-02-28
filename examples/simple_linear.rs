// use mlrs::dataset::Dataset;
// use mlrs::models::supervised::linear_regression::LinearRegression;
// use mlrs::training::Train;
use ndarray::array;

fn main() {
    //     // Simple dataset: y = 2x + 1
    let _features = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let _target = array![3.0, 6.0, 7.0, 10.0, 11.0];

    //     let dataset = Dataset::new(features, target);

    //     let mut model = LinearRegression::new();

    //     println!(
    //         "Before training: weight={}, bias={}",
    //         model.weight, model.bias
    //     );

    //     model.train(&dataset, 100, 0.01);

    //     println!(
    //         "After training: weight={}, bias={}",
    //         model.weight, model.bias
    //     );

    //     // Test prediction
    //     let test_x = 6.0;
    //     let prediction = model.predict(test_x);
    //     println!("Prediction for x={}: {}", test_x, prediction);
}
