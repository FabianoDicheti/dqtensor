mod data;
mod mlp;
mod optimizers;
mod f_not_linear;

use data::ingestion::DataFrame;
use mlp::dense::DenseLayer;
use optimizers::bp_optimizers::{SGD, Optimizer};
use f_not_linear::activation::ActivationFunction;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::HashMap;

fn main() {
    println!("================== IRIS MLP TEST ==================");

    // Carregar o dataset
    let mut df = DataFrame::from_file("./iris.csv").expect("Erro ao carregar CSV");
    df.shuffle_rows(5); // embaralhar os dados
    df.show_head(5);

    // Features e Target
    let features = df.extract_features().unwrap();
    let target_col = &df.columns[df.columns.len() - 1]; // Última coluna = target
    let targets = DataFrame::encode_column(target_col);
    let targets: Vec<Vec<f64>> = targets
        .iter()
        .map(|row| row.iter().map(|v| *v as f64).collect())
        .collect();

    // Split em treino e teste (80/20)
    let split_idx = (features.len() as f64 * 0.8) as usize;
    let (x_train, x_test) = features.split_at(split_idx);
    let (y_train, y_test) = targets.split_at(split_idx);

    // Definir arquitetura da MLP
    let input_size = 4;
    let hidden_size = 5;
    let output_size = 3;

    let mut dense1 = DenseLayer::new(
        input_size,
        hidden_size,
        ActivationFunction::ReLU,
        Box::new(SGD::new(0.01)),
        Box::new(SGD::new(0.01)),
    );

    let mut dense2 = DenseLayer::new(
        hidden_size,
        output_size,
        ActivationFunction::Softmax(output_size),
        Box::new(SGD::new(0.01)),
        Box::new(SGD::new(0.01)),
    );

    // Loop de treinamento
    let epochs = 500;
    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut correct = 0;

        for (x, y) in x_train.iter().zip(y_train.iter()) {
            // Forward
            let out1 = dense1.forward(x);
            let out2 = dense2.forward(&out1);

            // Calcular erro (MSE simplificado)
            let mut d_loss_d_out = vec![0.0; output_size];
            for i in 0..output_size {
                let target = y[i];
                let error = out2[i] - target;
                d_loss_d_out[i] = error;
                total_loss += error.powi(2);
            }

            // Acuracia
            let pred = argmax(&out2);
            let real = argmax(&y);
            if pred == real {
                correct += 1;
            }

            // Backward
            let grad2 = dense2.backward(&out1, &out2, &d_loss_d_out);
            let _grad1 = dense1.backward(x, &out1, &grad2);
        }

        if epoch % 100 == 0 {
            println!(
                "Epoch {} -> Loss: {:.4} | Acurácia: {:.2}%",
                epoch,
                total_loss / x_train.len() as f64,
                correct as f64 / x_train.len() as f64 * 100.0
            );
        }
    }

    println!("Treinamento finalizado.");

    println!("\n========= AVALIAÇÃO NO TESTE =========");

    let mut y_true = vec![];
    let mut y_pred = vec![];

    for (x, y) in x_test.iter().zip(y_test.iter()) {
        let out1 = dense1.forward(x);
        let out2 = dense2.forward(&out1);

        let pred = argmax(&out2);
        let real = argmax(&y);

        y_true.push(real);
        y_pred.push(pred);
    }

    let metrics = classification_report(&y_true, &y_pred, output_size);
    println!("{}", metrics);
}

/// Indice do maior valor (argmax)
fn argmax(v: &Vec<f64>) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0
}

/// Precision, Recall e F1
fn classification_report(y_true: &Vec<usize>, y_pred: &Vec<usize>, num_classes: usize) -> String {
    let mut tp = vec![0.0; num_classes];
    let mut fp = vec![0.0; num_classes];
    let mut fn_ = vec![0.0; num_classes];

    for (true_label, pred_label) in y_true.iter().zip(y_pred.iter()) {
        if true_label == pred_label {
            tp[*true_label] += 1.0;
        } else {
            fp[*pred_label] += 1.0;
            fn_[*true_label] += 1.0;
        }
    }

    let mut report = String::from("\nClasse | Precision | Recall | F1-score\n");
    report.push_str("-----------------------------------------\n");

    for c in 0..num_classes {
        let precision = if tp[c] + fp[c] == 0.0 {
            0.0
        } else {
            tp[c] / (tp[c] + fp[c])
        };
        let recall = if tp[c] + fn_[c] == 0.0 {
            0.0
        } else {
            tp[c] / (tp[c] + fn_[c])
        };
        let f1 = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * (precision * recall) / (precision + recall)
        };

        report.push_str(&format!(
            "   {}    |   {:.2}     |  {:.2}  |  {:.2}\n",
            c, precision, recall, f1
        ));
    }

    report
}
