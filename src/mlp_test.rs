use dqtensor::data::ingestion::{DataFrame, normalize_min_max};
use dqtensor::mlp::dense::DenseLayer;
use dqtensor::f_not_linear::activation::ActivationFunction;
use dqtensor::optimizers::bp_optimizers::Adam;
use dqtensor::loss_functions::loss::cross_entropy;

use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::HashMap;
use std::error::Error;

const HIDDEN1_SIZE: usize = 256;
const HIDDEN2_SIZE: usize = 128;
const OUTPUT_SIZE: usize = 3;
const INPUT_SIZE: usize = 4;


pub fn run_experiment() -> Result<(), Box<dyn Error>> {
    let (features, labels) = load_iris_dataset("iris.csv");
    println!("Dataset loaded with {} examples.\n", features.len());

    let (train_x, train_y, test_x, test_y) = train_test_split(&features, &labels, 0.8);

    let architectures = vec![
        ("Homogeneous: Tanh , Tanh", vec![ActivationFunction::Tanh; HIDDEN1_SIZE]),
        ("Heterogeneous: ReLU/Tanh", (0..HIDDEN1_SIZE)
            .map(|i| if i % 2 == 0 { ActivationFunction::Relu } else { ActivationFunction::Tanh })
            .collect()),
        ("Heterogeneous: Sigmoid/Tanh", (0..HIDDEN1_SIZE)
            .map(|i| if i % 2 == 0 { ActivationFunction::Sigmoid } else { ActivationFunction::Tanh })
            .collect()),
        ("Heterogeneous: ReLU/Tanh/Sigmoid", (0..HIDDEN1_SIZE)
            .map(|i| match i % 3 {
                0 => ActivationFunction::Relu,
                1 => ActivationFunction::Tanh,
                _ => ActivationFunction::Sigmoid
            })
            .collect()),
    ];

    for (name, activations_layer1) in architectures {
        println!("=== Architecture: {} ===", name);

        // --- Layer 1: Heterogeneous --- //
        let mut layer1 = DenseLayer::new_heterogeneous(
            INPUT_SIZE,
            activations_layer1.clone(),
            Box::new(Adam::new(0.001, HIDDEN1_SIZE * INPUT_SIZE)),
            Box::new(Adam::new(0.001, HIDDEN1_SIZE)),
        );

        // --- Layer 2: Homogeneous --- //
        let mut layer2 = DenseLayer::new(
            HIDDEN1_SIZE,
            HIDDEN2_SIZE,
            ActivationFunction::Tanh,
            Box::new(Adam::new(0.001, HIDDEN2_SIZE * HIDDEN1_SIZE)),
            Box::new(Adam::new(0.001, HIDDEN2_SIZE)),
        );

        // --- Output Layer --- //
        let mut output_layer = DenseLayer::new(
            HIDDEN2_SIZE,
            OUTPUT_SIZE,
            ActivationFunction::Sigmoid,
            Box::new(Adam::new(0.001, OUTPUT_SIZE * HIDDEN2_SIZE)),
            Box::new(Adam::new(0.001, OUTPUT_SIZE)),
        );

        // --- Training Loop --- //
        let num_epochs = 35;
        for _ in 0..num_epochs {
            for (x, y) in train_x.iter().zip(train_y.iter()) {
                let mut y_onehot = vec![0.0; 3];
                y_onehot[*y] = 1.0;

                let out1 = layer1.forward(x);
                let out2 = layer2.forward(&out1);
                let out3 = output_layer.forward(&out2);

                let _loss = cross_entropy(&y_onehot, &out3);

                let grad_loss: Vec<f64> = out3.iter().zip(&y_onehot).map(|(p, t)| p - t).collect();
                let grad_out2 = output_layer.backward(&grad_loss);
                let grad_out1 = layer2.backward(&grad_out2);
                let _ = layer1.backward(&grad_out1);

                output_layer.update();
                layer2.update();
                layer1.update();
            }
        }

        // --- Evaluation --- //
        let (train_acc, train_conf) = evaluate_with_confusion(&mut layer1, &mut layer2, &mut output_layer, &train_x, &train_y);
        let (test_acc, test_conf) = evaluate_with_confusion(&mut layer1, &mut layer2, &mut output_layer, &test_x, &test_y);

        println!(
            "Train Accuracy: {:.2}% | Test Accuracy: {:.2}%",
            train_acc * 100.0,
            test_acc * 100.0
        );

        println!("Confusion Matrix (Train):");
        print_confusion_matrix(&train_conf);

        println!("Confusion Matrix (Test):");
        print_confusion_matrix(&test_conf);

        println!("---------------------------------------------------------\n");
    }

    Ok(())
}


fn load_iris_dataset(path: &str) -> (Vec<Vec<f64>>, Vec<usize>) {
    let df = DataFrame::from_file(path).expect("Failed to read iris.csv");
    let mut features = df.extract_features().expect("Failed to extract features");

    let label_idx = df.columns.len() - 1;
    let raw_labels = &df.columns[label_idx];
    let labels: Vec<usize> = raw_labels
        .iter()
        .map(|class| {
            let class_norm = class.to_lowercase();
            match class_norm.as_str() {
                "iris-setosa" | "setosa" => 0,
                "iris-versicolor" | "versicolor" => 1,
                "iris-virginica" | "virginica" => 2,
                _ => panic!("Unknown class: {}", class),
            }
        })
        .collect();

    normalize_min_max(&mut features);
    (features, labels)
}


fn train_test_split(
    x: &Vec<Vec<f64>>,
    y: &Vec<usize>,
    train_ratio: f64,
) -> (Vec<Vec<f64>>, Vec<usize>, Vec<Vec<f64>>, Vec<usize>) {
    let total = x.len();
    let train_size = (total as f64 * train_ratio).round() as usize;

    let mut indices: Vec<usize> = (0..x.len()).collect();
    indices.shuffle(&mut thread_rng());

    let (train_idx, test_idx) = indices.split_at(train_size);
    let train_x = train_idx.iter().map(|&i| x[i].clone()).collect();
    let train_y = train_idx.iter().map(|&i| y[i]).collect();
    let test_x = test_idx.iter().map(|&i| x[i].clone()).collect();
    let test_y = test_idx.iter().map(|&i| y[i]).collect();

    (train_x, train_y, test_x, test_y)
}


fn evaluate_with_confusion(
    layer1: &mut DenseLayer,
    layer2: &mut DenseLayer,
    output: &mut DenseLayer,
    x: &Vec<Vec<f64>>,
    y: &Vec<usize>,
) -> (f64, [[u32; 3]; 3]) {
    let mut correct = 0;
    let mut confusion = [[0u32; 3]; 3];

    for (input, true_class) in x.iter().zip(y.iter()) {
        let o1 = layer1.forward(input);
        let o2 = layer2.forward(&o1);
        let output_vals = output.forward(&o2);

        let predicted = output_vals
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        if predicted == *true_class {
            correct += 1;
        }

        confusion[*true_class][predicted] += 1;
    }

    (correct as f64 / x.len() as f64, confusion)
}


fn print_confusion_matrix(matrix: &[[u32; 3]; 3]) {
    println!("       Predicted");
    println!("       0   1   2");
    for (i, row) in matrix.iter().enumerate() {
        println!("Actual {}  {:3} {:3} {:3}", i, row[0], row[1], row[2]);
    }
}
