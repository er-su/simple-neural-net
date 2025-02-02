use mnist::*;
use ndarray::prelude::*;

pub fn read_mnist_images(train: bool) -> Vec<Vec<f64>> {
    let Mnist {
        trn_img,
        tst_img,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(60_000)
        .validation_set_length(0)
        .test_set_length(10_000)
        .finalize();

    if train {
        let training_data = Array2::from_shape_vec((60_000, 784), trn_img)
                            .expect("Error converting training images to Array2 Struct")
                            .map(|x| *x as f64 / 256.0);
        
        let training_data_vec: Vec<Vec<f64>> = training_data
            .axis_iter(Axis(0))
            .map(|row| row.to_vec())
            .collect();

        training_data_vec
    }

    else {
        let testing_data = Array2::from_shape_vec((10_000, 784), tst_img)
            .expect("Error converting test images to Array2 struct")
            .map(|x| *x as f64 / 256.0);

        let testing_data_vec: Vec<Vec<f64>> = testing_data
            .axis_iter(Axis(0))
            .map(|row| row.to_vec())
            .collect();

        testing_data_vec
    }    
    
}

pub fn read_mnist_labels(train: bool) -> Vec<usize> {
    let Mnist {
        trn_lbl,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(60_000)
        .validation_set_length(0)
        .test_set_length(10_000)
        .finalize();

    if train {
        let training_labels = Array2::from_shape_vec((60_000, 1), trn_lbl)
            .expect("Error converting training labels to Array2 struct");

        let training_labels_vec: Vec<usize> = training_labels
            .iter()
            .map(|x| (*x) as usize)
            .collect();

        training_labels_vec
        
    }

    else {
        let testing_labels = Array2::from_shape_vec((10_000, 1), tst_lbl)
            .expect("Error converting testing labels to Array2 struct");

        let testing_labels_vec: Vec<usize> = testing_labels
            .iter()
            .map(|x| (*x) as usize)
            .collect();

        testing_labels_vec
    }
}