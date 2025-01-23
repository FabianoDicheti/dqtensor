pub enum LossFuncion {
    MeanSquaredError,
    CrossEntropy,
    MeanAbsoluteError,
    HuberLoss(f64),
}

impl LossFuncion {

    pub fn calculate( &self, predictions: &[f64], targets: &[f64]) -> f64 {
        match self {
            LossFuncion::MeanSquaredError => Self::mean_squared_error(predictions, targets),
            LossFuncion::CrossEntropy => Self::cross_entropy(predictions, targets),
            LossFuncion::MeanAbsoluteError => Self::mean_absolute_error(predictions, targets),
            LossFuncion::HuberLoss(delta) => Self::huber_loss(predictions, targets, *delta),
        }
    }

    fn mean_squared_error(predictions: &[f64], targets: &[f64]) -> f64 {
        let n = predictions.len();
        predictions.iter()
            .zip(targets.iter())
            .map(|(&p, &t)| (p-t).powi(2))
            .sum::<f64>()
            / n as f64
    }

    fn cross_entropy(predictions: &[f64], targets: &[f64]) -> f64 {
        predictions.iter()
            .zip(targets.iter())
            .map(|(&p, &t)| {
                if t == 1.0 {
                    -p.ln()
                } else if t == 0.0 {
                    -(1.0 -p).ln()
                } else {
                    panic!("Targets for Cross-Entropy must be (0 or 1)!");
                }
            })
            .sum()
    }

    fn mean_absolute_error(predictions: &[f64], targets: &[f64]) -> f64 {
        let n = predictions.len();
        predictions.iter()
            .zip(targets.iter())
            .map(|(&p, &t)| (p-t).abs())
            .sum::<f64>()
            / n as f64
    }

    fn huber_loss(predictions: &[f64], targets: &[f64], delta: f64) -> f64 { // delta define o limite entre os comportamentos quadráticos e linear
        let n = predictions.len();
        predictions
            .iter()
            .zip(targets.iter()) // zip é pra iterar nos dois, no caso ta zipando o predictions com o targets, pra iterar sobre os dois ao mesmo tempo
            .map(|(&p, &t)|{ // p é o valor do iterador que estiver em predictions e t é o valor do mesmo indice no targets, a notação |(&p, &t)| é pra remover a referência e pegar os valores como f64 e não como ref.
                let error = (p - t).abs();
                if error <= delta { // se o erro for menor que o limiar vai na quadratica, se for maior vai na linear
                    0.5 * error.powi(2)
                } else {
                    delta * (error - 0.5 * delta)
                }
            })
            .sum::<f64>() // soma todas as perdas calculadas para cara par p e t
            / n as f64 // retorna a média dividindo a soma pelo n total
    }
}



//huber loss
//log-cosh loss
//quantile loss
//kullback-leibler divergence
//focal loss
//hinge loss
//categorical hinge loss
//IoU loss
//Dice loss
//Triplet Loss
