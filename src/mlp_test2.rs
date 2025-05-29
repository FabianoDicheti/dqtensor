use dqtensor::mlp::dense::DenseLayer;
use dqtensor::f_not_linear::activation::ActivationFunction;
use dqtensor::optimizers::bp_optimizers::{Adam, Optimizer}; 
use dqtensor::data::ingestion::{DataFrame, normalize_min_max};
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::HashMap;
use std::error::Error;

const HIDDEN1_SIZE: usize = 512;
const HIDDEN2_SIZE: usize = 512;
const HIDDEN3_SIZE: usize = 256;
const OUTPUT_SIZE: usize = 6;
const INPUT_SIZE: usize = 11;

//  Helper //

fn softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return vec![];
    }
    let max_logit = logits.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum_exps: f64 = exps.iter().sum();
    exps.iter().map(|&exp| exp / sum_exps).collect()
}


fn cross_entropy_loss(probs: &[f64], target_one_hot: &[f64]) -> f64 {
    if probs.len() != target_one_hot.len() || probs.is_empty() {
        return 0.0; 
    }

    let epsilon = 1e-12;
    -target_one_hot.iter().zip(probs.iter())
        .map(|(t, p)| t * (p + epsilon).ln())
        .sum::<f64>()
}


fn cross_entropy_softmax_gradient(probs: &[f64], target_one_hot: &[f64]) -> Vec<f64> {
    if probs.len() != target_one_hot.len() {
        return vec![0.0; probs.len()];
    }
    probs.iter().zip(target_one_hot.iter())
        .map(|(p, t)| p - t)
        .collect()
}

// EXPERIMENT //

pub fn run_experiment_wine() -> Result<(), Box<dyn Error>> {
    let (features, labels, label_names) =
        load_wine_dataset("wine_dataset/winequality-red.csv")?;
    println!("Dataset loaded with {} examples.", features.len());
    println!("Found classes: {:?} ({} classes)\n", label_names, label_names.len());

    if OUTPUT_SIZE != label_names.len() {
        return Err(format!(
            "OUTPUT_SIZE ({}) mismatch with found labels ({})",
            OUTPUT_SIZE,
            label_names.len()
        ).into());
    }

    let (train_x, train_y, test_x, test_y) = train_test_split(&features, &labels, 0.8);

    let architectures = vec![
        (
            "Heterogeneous - Mix1 (Relu/Tanh/Sig/ELU)",
            mix_activations(HIDDEN1_SIZE), //  ELU 
            mix_activations2(HIDDEN2_SIZE),
            mix_activations(HIDDEN3_SIZE), // ELU 
        ),
        (
            "Heterogeneous - Mix2 (Relu/Tanh/ELU/Sig)",
            mix_activations2(HIDDEN1_SIZE),
            mix_activations2b(HIDDEN2_SIZE), // ELU
            mix_activations2(HIDDEN3_SIZE),
        ),
        (
            "Homogeneous - Relu",
            vec![ActivationFunction::Relu; HIDDEN1_SIZE],
            vec![ActivationFunction::Relu; HIDDEN2_SIZE],
            vec![ActivationFunction::Relu; HIDDEN3_SIZE],
        ),
        (
            "Homogeneous - Tanh",
            vec![ActivationFunction::Tanh; HIDDEN1_SIZE],
            vec![ActivationFunction::Tanh; HIDDEN2_SIZE],
            vec![ActivationFunction::Tanh; HIDDEN3_SIZE],
        ),
    ];

    println!("\n\n=== TEST 2: MLP WINE DATASET ===\n");

    for (name, act1, act2, act3) in architectures {
        println!("=== Architecture: {} ===", name);
        print_architecture(&act1, &act2, &act3);

        let mut layer1 = DenseLayer::new_heterogeneous(
            INPUT_SIZE,
            act1, 
            Box::new(Adam::new(0.001, INPUT_SIZE * HIDDEN1_SIZE)) as Box<dyn Optimizer>,
            Box::new(Adam::new(0.001, HIDDEN1_SIZE)) as Box<dyn Optimizer>,
        );

        let mut layer2 = DenseLayer::new_heterogeneous(
            HIDDEN1_SIZE,
            act2, 
            Box::new(Adam::new(0.001, HIDDEN1_SIZE * HIDDEN2_SIZE)) as Box<dyn Optimizer>,
            Box::new(Adam::new(0.001, HIDDEN2_SIZE)) as Box<dyn Optimizer>,
        );

        let mut layer3 = DenseLayer::new_heterogeneous(
            HIDDEN2_SIZE,
            act3,
            Box::new(Adam::new(0.001, HIDDEN2_SIZE * HIDDEN3_SIZE)) as Box<dyn Optimizer>,
            Box::new(Adam::new(0.001, HIDDEN3_SIZE)) as Box<dyn Optimizer>,
        );

        let mut output_layer = DenseLayer::new(
            HIDDEN3_SIZE,
            OUTPUT_SIZE,
            ActivationFunction::Linear, 
            Box::new(Adam::new(0.001, HIDDEN3_SIZE * OUTPUT_SIZE)) as Box<dyn Optimizer>,
            Box::new(Adam::new(0.001, OUTPUT_SIZE)) as Box<dyn Optimizer>,
        );

        let num_epochs = 35;

        for epoch in 0..num_epochs {
            let mut epoch_loss = 0.0;
            let mut indices: Vec<usize> = (0..train_x.len()).collect();
            indices.shuffle(&mut thread_rng());

            for &i in &indices {
                let x = &train_x[i];
                let y = train_y[i];
                let mut y_onehot = vec![0.0; OUTPUT_SIZE];
                y_onehot[y] = 1.0;

                // Forward
                let out1 = layer1.forward(x);
                let out2 = layer2.forward(&out1);
                let out3 = layer3.forward(&out2);
                let logits = output_layer.forward(&out3);
                let probs = softmax(&logits); //manually

                let loss = cross_entropy_loss(&probs, &y_onehot);
                epoch_loss += loss;

                // Backward 
                let grad_loss = cross_entropy_softmax_gradient(&probs, &y_onehot);

                let grad_out4 = output_layer.backward(&grad_loss);
                let grad_out3 = layer3.backward(&grad_out4);
                let grad_out2 = layer2.backward(&grad_out3);
                let _ = layer1.backward(&grad_out2);

                // Update weights
                output_layer.update();
                layer3.update();
                layer2.update();
                layer1.update();
            }
            if (epoch + 1) % 5 == 0 {
                 println!("  Epoch {}/{}, Avg Loss: {:.6}", epoch + 1, num_epochs, epoch_loss / train_x.len() as f64);
            }
        }

        let (train_acc, train_conf) =
            evaluate_with_confusion(&mut layer1, &mut layer2, &mut layer3, &mut output_layer, &train_x, &train_y);
        let (test_acc, test_conf) =
            evaluate_with_confusion(&mut layer1, &mut layer2, &mut layer3, &mut output_layer, &test_x, &test_y);

        println!(
            "Train Accuracy: {:.2}% | Test Accuracy: {:.2}%",
            train_acc * 100.0,
            test_acc * 100.0
        );


        println!("Confusion Matrix (Train):");
        print_confusion_matrix(&train_conf, &label_names);

        println!("Confusion Matrix (Test):");
        print_confusion_matrix(&test_conf, &label_names);

        println!("---------------------------------------------------------\n");
    }


    Ok(())
}

fn evaluate_with_confusion(
    layer1: &mut DenseLayer,
    layer2: &mut DenseLayer,
    layer3: &mut DenseLayer,
    output_layer: &mut DenseLayer,
    x_set: &Vec<Vec<f64>>,
    y_set: &Vec<usize>,
) -> (f64, HashMap<(usize, usize), usize>) {
    let mut confusion = HashMap::new();

    let predictions: Vec<usize> = x_set
        .iter()
        .map(|x| {
            let out1 = layer1.forward(x);
            let out2 = layer2.forward(&out1);
            let out3 = layer3.forward(&out2);
            let logits = output_layer.forward(&out3);
            let probs = softmax(&logits);
            argmax(&probs)
        })
        .collect();

    for (pred, true_label) in predictions.iter().zip(y_set.iter()) {
        *confusion.entry((*true_label, *pred)).or_insert(0) += 1;
    }

    let correct = predictions
        .iter()
        .zip(y_set.iter())
        .filter(|(pred, true_label)| pred == true_label)
        .count();

    let acc = correct as f64 / x_set.len() as f64;
    (acc, confusion)
}

fn print_architecture(
    act1: &Vec<ActivationFunction>,
    act2: &Vec<ActivationFunction>,
    act3: &Vec<ActivationFunction>,
) {
    let count1 = count_activations(act1);
    let count2 = count_activations(act2);
    let count3 = count_activations(act3);

    println!("  ## Hidden 1: {:?}", count1);
    println!("  ## Hidden 2: {:?}", count2);
    println!("  ## Hidden 3: {:?}", count3);
    println!("  ## Output Layer: Linear + Softmax ({} classes)\n", OUTPUT_SIZE);
}

fn count_activations(acts: &Vec<ActivationFunction>) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for act in acts {
        let label = format!("{:?}", act);
        *counts.entry(label).or_insert(0) += 1;
    }
    counts
}

fn print_confusion_matrix(matrix: &HashMap<(usize, usize), usize>, labels: &Vec<String>) {
    print!("True + / Pred ##\t");
    for l in labels {
        print!("{}\t", l);
    }
    println!();

    for (i, true_label) in labels.iter().enumerate() {
        print!("       {}\t", true_label);
        for (j, _) in labels.iter().enumerate() {
            let count = matrix.get(&(i, j)).unwrap_or(&0);
            print!("{}\t", count);
        }
        println!();
    }
    println!();
}

fn load_wine_dataset(path: &str) -> Result<(Vec<Vec<f64>>, Vec<usize>, Vec<String>), Box<dyn Error>> {
    let df = DataFrame::from_file(path)?;
    // df.show_head(5); 

    let mut features = df.extract_features()?;
    normalize_min_max(&mut features);

    let labels_col = &df.columns[df.columns.len() - 1];

    let mut unique_labels: Vec<String> = labels_col.clone();
    unique_labels.sort_unstable();
    unique_labels.dedup();

    let label_map: HashMap<String, usize> = unique_labels
        .iter()
        .enumerate()
        .map(|(i, label)| (label.clone(), i))
        .collect();

    let labels: Vec<usize> = labels_col
        .iter()
        .map(|v| *label_map.get(v).expect("Label not found in map"))
        .collect();

    Ok((features, labels, unique_labels))
}

fn train_test_split(
    x: &Vec<Vec<f64>>,
    y: &Vec<usize>,
    train_ratio: f64,
) -> (Vec<Vec<f64>>, Vec<usize>, Vec<Vec<f64>>, Vec<usize>) {
    let mut indices: Vec<usize> = (0..x.len()).collect();
    indices.shuffle(&mut thread_rng());

    let train_size = (x.len() as f64 * train_ratio).round() as usize;
    let (train_idx, test_idx) = indices.split_at(train_size);

    let train_x = train_idx.iter().map(|&i| x[i].clone()).collect();
    let train_y = train_idx.iter().map(|&i| y[i]).collect();
    let test_x = test_idx.iter().map(|&i| x[i].clone()).collect();
    let test_y = test_idx.iter().map(|&i| y[i]).collect();

    (train_x, train_y, test_x, test_y)
}

fn argmax(v: &Vec<f64>) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// Mix activations 
fn mix_activations(size: usize) -> Vec<ActivationFunction> {
    let funcs = vec![
        ActivationFunction::Relu,
        ActivationFunction::Tanh,
        ActivationFunction::Sigmoid,
        ActivationFunction::ELU, // *
    ];
    (0..size)
        .map(|i| funcs[i % funcs.len()].clone())
        .collect()
}

fn mix_activations2(size: usize) -> Vec<ActivationFunction> {
    let funcs = vec![ActivationFunction::Relu, ActivationFunction::Tanh];
    (0..size)
        .map(|i| funcs[i % funcs.len()].clone())
        .collect()
}

fn mix_activations2b(size: usize) -> Vec<ActivationFunction> {
    let funcs = vec![ActivationFunction::ELU, ActivationFunction::Sigmoid]; // *
    (0..size)
        .map(|i| funcs[i % funcs.len()].clone())
        .collect()
}

