// use csv::ReaderBuilder;
// use log::error;
// use ndarray::{Array1, Array2};
// use std::error::Error;
// use std::fs::File;

pub mod raw_dataset;

// pub struct Dataset {
//     features: Vec<String>,
//     target: String,
//     x_train: Array2<f64>,
//     y_train: Option<Array1<f64>>,
//     x_test: Option<Array2<f64>>,
//     y_test: Option<Array1<f64>>,
//     x_val: Option<Array2<f64>>,
//     y_val: Option<Array1<f64>>,
// }

// impl Dataset {
//     pub fn new(
//         filepath: String,
//         delimiter: char,
//         has_headers: bool,
//     ) -> Result<Self, Box<dyn Error>> {
//         let file_reader = match File::open(&filepath) {
//             Ok(file) => file,
//             Err(e) => {
//                 error!("Error opening file in {}: {}", filepath, e);
//                 return Err(Box::new(e));
//             }
//         };
//         let _csv_reader = ReaderBuilder::new()
//             .delimiter(delimiter as u8)
//             .has_headers(has_headers)
//             .from_reader(file_reader);

//         Ok(Self {
//             features: Vec::new(),
//             target: String::new(),
//             x_train: Array2::zeros((0, 0)),
//             y_train: None,
//             x_test: None,
//             y_test: None,
//             x_val: None,
//             y_val: None,
//         })
//     }
// }
