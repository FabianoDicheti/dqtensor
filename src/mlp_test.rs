use dqtensor::mlp::dense::DenseLayer;
use dqtensor::f_not_linear::activation::ActivationFunction;
use dqtensor::optimizers::bp_optimizers::Adam;
use dqtensor::data::ingestion::DataFrame;
use dqtensor::loss_functions::loss_functions::cross_entropy;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::HashMap;
use std::error::Error;

const HIDDEN1_SIZE: usize = 256;
const HIDDEN2_SIZE: usize = 128;
const OUTPUT_SIZE: usize = 3;
const INPUT_SIZE: usize = 4;

pub fn run_experiment() -> Result<(), Box<dyn Error>> {
    let (features, labels) = load_iris_with_dataframe("iris.csv");
    println!("Dataset carregado com {} exemplos.\n", features.len());

    let (train_x, train_y, test_x, test_y) = train_test_split(&features, &labels, 0.8);

    let architectures = vec![
        ("Homogênea - ReLU", vec![ActivationFunction::ReLU; HIDDEN1_SIZE], vec![ActivationFunction::ReLU; HIDDEN2_SIZE]),
        ("Homogênea - Tanh", vec![ActivationFunction::Tanh; HIDDEN1_SIZE], vec![ActivationFunction::Tanh; HIDDEN2_SIZE]),
        ("Heterogênea - Mix1", mix_activations(HIDDEN1_SIZE), mix_activations(HIDDEN2_SIZE)),
        ("Heterogênea - Mix2", mix_activations2(HIDDEN1_SIZE), mix_activations2(HIDDEN2_SIZE)),
    ];

    for (name, act1, act2) in architectures {
        println!("=== Arquitetura: {} ===", name);
        print_architecture(&act1, &act2);

        let mut layer1 = DenseLayer::new(
            INPUT_SIZE,
            HIDDEN1_SIZE,
            act1,
            Box::new(Adam::new(0.001, HIDDEN1_SIZE * INPUT_SIZE)),
            Box::new(Adam::new(0.001, HIDDEN1_SIZE)),
        );

        let mut layer2 = DenseLayer::new(
            HIDDEN1_SIZE,
            HIDDEN2_SIZE,
            act2,
            Box::new(Adam::new(0.001, HIDDEN2_SIZE * HIDDEN1_SIZE)),
            Box::new(Adam::new(0.001, HIDDEN2_SIZE)),
        );

        let mut output_layer = DenseLayer::new(
            HIDDEN2_SIZE,
            OUTPUT_SIZE,
            vec![ActivationFunction::Softmax(OUTPUT_SIZE); OUTPUT_SIZE],
            Box::new(Adam::new(0.001, OUTPUT_SIZE * HIDDEN2_SIZE)),
            Box::new(Adam::new(0.001, OUTPUT_SIZE)),
        );

        let num_epochs = 55;

        for _ in 0..num_epochs {
            for (x, y) in train_x.iter().zip(train_y.iter()) {
                let mut y_onehot = vec![0.0; 3];
                y_onehot[*y] = 1.0;

                let out1 = layer1.forward(x);
                let out2 = layer2.forward(&out1);
                let out3 = output_layer.forward(&out2);

                let _loss = cross_entropy(&y_onehot, &out3);

                let grad_loss = out3.iter().zip(&y_onehot).map(|(p, t)| p - t).collect();
                let grad_out2 = output_layer.backward(&grad_loss);
                let grad_out1 = layer2.backward(&grad_out2);
                let _ = layer1.backward(&grad_out1);

                output_layer.update(0.001);
                layer2.update(0.001);
                layer1.update(0.001);
            }
        }

        let (train_acc, train_conf) = evaluate_with_confusion(&mut layer1, &mut layer2, &mut output_layer, &train_x, &train_y);
        let (test_acc, test_conf) = evaluate_with_confusion(&mut layer1, &mut layer2, &mut output_layer, &test_x, &test_y);

        println!(
            "Acurácia Treino: {:.2}% | Acurácia Teste: {:.2}%",
            train_acc * 100.0,
            test_acc * 100.0
        );

        println!("Matriz de Confusão (Treino):");
        print_confusion_matrix(&train_conf);

        println!("Matriz de Confusão (Teste):");
        print_confusion_matrix(&test_conf);

        println!("---------------------------------------------------------\n");
    }

    Ok(())
}

/// Avaliação
fn evaluate_with_confusion(
    layer1: &mut DenseLayer,
    layer2: &mut DenseLayer,
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
            let out3 = output_layer.forward(&out2);
            argmax(&out3)
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

/// Mostrar arquitetura
fn print_architecture(act1: &Vec<ActivationFunction>, act2: &Vec<ActivationFunction>) {
    let count1 = count_activations(act1);
    let count2 = count_activations(act2);

    println!("  → Camada Oculta 1: {:?}", count1);
    println!("  → Camada Oculta 2: {:?}", count2);
    println!("  → Camada Saída: Softmax (3 classes)\n");
}

/// Contar ativações
fn count_activations(acts: &Vec<ActivationFunction>) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for act in acts {
        let label = format!("{:?}", act);
        *counts.entry(label).or_insert(0) += 1;
    }
    counts
}

/// Print Matriz de Confusão
fn print_confusion_matrix(matrix: &HashMap<(usize, usize), usize>) {
    println!("True ↓ / Pred →\t 0\t 1\t 2");
    for true_label in 0..3 {
        print!("       {}\t", true_label);
        for pred_label in 0..3 {
            let count = matrix.get(&(true_label, pred_label)).unwrap_or(&0);
            print!("{}\t", count);
        }
        println!();
    }
    println!();
}

/// Funções auxiliares
fn load_iris_with_dataframe(path: &str) -> (Vec<Vec<f64>>, Vec<usize>) {
    let df = DataFrame::from_file(path).expect("Falha ao carregar o CSV");
    df.show_head(5);

    let features = df.extract_features().expect("Erro nas features");

    let labels_col = &df.columns[df.columns.len() - 1];
    let labels: Vec<usize> = labels_col
        .iter()
        .map(|v| match v.trim().to_lowercase().as_str() {
            "setosa" | "iris-setosa" => 0,
            "versicolor" | "iris-versicolor" => 1,
            "virginica" | "iris-virginica" => 2,
            other => panic!("Label desconhecido: {}", other),
        })
        .collect();

    (features, labels)
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
        .unwrap()
}

fn mix_activations(size: usize) -> Vec<ActivationFunction> {
    let funcs = vec![
        ActivationFunction::ReLU,
        ActivationFunction::Tanh,
        ActivationFunction::Sigmoid,
        ActivationFunction::Softplus,
    ];
    (0..size)
        .map(|i| funcs[i % funcs.len()].clone())
        .collect()
}


fn mix_activations2(size: usize) -> Vec<ActivationFunction> {
    let funcs = vec![
        ActivationFunction::ReLU,
        ActivationFunction::Tanh,
    ];
    (0..size)
        .map(|i| funcs[i % funcs.len()].clone())
        .collect()
}
