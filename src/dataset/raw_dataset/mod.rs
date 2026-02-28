use csv::{Reader, ReaderBuilder};
use log::error;
use rayon;
use std::{error::Error, fs::File};

enum ColumnType {
    Int(i32),
    Float(f32),
    String(String),
    Boolean(bool),
}

fn assign_type(filepath: String, delimiter: char, has_headers: bool) {
    let csv_reader = load_csv(filepath, delimiter, has_headers);
    // Check the first row of the dataset, if the value contains only numbers, assign Float or Boolean.
    // Make 6 threads and make them randomly select a row and return the datatype acocrding to
    // them. Then based on that we can assign the ColumnType.
    // If the 1st row has string alphabets, then return String and there is no need to spawn threads. If the 1st row says,
    // not string, then we need to assign either Int, Float or Boolean. (Maybe the column values
    // are true or false in string). Now you know what to do. "yes" and "no" are common as well.
}

pub fn load_csv(
    filepath: String,
    delimiter: char,
    has_headers: bool,
) -> Result<Reader<File>, Box<dyn Error>> {
    let file_reader: File = match File::open(&filepath) {
        Ok(file) => file,
        Err(err) => {
            error!("Error opening at location: {}: {}", filepath, err);
            return Err(Box::new(err));
        }
    };
    let csv_reader = ReaderBuilder::new()
        .delimiter(delimiter as u8)
        .has_headers(has_headers)
        .from_reader(file_reader);

    Ok(csv_reader)
}

pub fn label_encoder() -> Result<(), rayon::ThreadPoolBuildError> {
    std::thread::scope(|scope| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .spawn_handler(|thread| {
                let mut builder = std::thread::Builder::new();
                if let Some(name) = thread.name() {
                    builder = builder.name(name.to_string());
                }
                if let Some(size) = thread.stack_size() {
                    builder = builder.stack_size(size);
                }
                builder.spawn_scoped(scope, || thread.run())?;
                Ok(())
            })
            .build()?;
        pool.install(|| println!("Hello from kartik!"));
        Ok(())
    })
}
