use std::f64;

pub enum LossFuncion {
    MeanSquaredError,
    CrossEntropy,
    MeanAbsoluteError,
    HuberLoss(f64),
    LogCoshLoss,
    QuantileLoss(f64),
    KLDivergence,
}

impl LossFuncion {

    pub fn calculate( &self, predictions: &[f64], targets: &[f64]) -> f64 {
        match self {
            LossFuncion::MeanSquaredError => Self::mean_squared_error(predictions, targets),
            LossFuncion::CrossEntropy => Self::cross_entropy(predictions, targets),
            LossFuncion::MeanAbsoluteError => Self::mean_absolute_error(predictions, targets),
            LossFuncion::HuberLoss(delta) => Self::huber_loss(predictions, targets, *delta),
            LossFuncion::LogCoshLoss => Self::log_cosh_loss(predictions, targets),
            LossFuncion::QuantileLoss(quantile) => Self::quantinle_loss(predictions, targets, *quantile),
            LossFuncion::KLDivergence => Self::kl_divergence(predictions, targets),

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

    fn log_cosh_loss(predictions: &[f64], targets: &[f64]) -> f64{
        
        assert_eq!(
            predictions.len(),
            targets.len(),
            "predictions size != targets size (len)"
        );

        predictions
            .iter()
            .zip(targets.iter()) // zipa pra iterar ao mesmo tempo nos dois
            .map(|(&p, &t)|{
                let diff = p - t; // std::f64::consts::LN_2, é a constante padrão pra o logaritmo natual de 2 (0.6931471.....) 
                (diff.exp()+(-diff).exp()).ln() - std::f64::consts::LN_2//.ln é o logaritmo natural .exp é expoente de euler,
            })
            .sum::<f64>()
            / predictions.len() as f64
    }

    fn quantinle_loss(predictions: &[f64], targets: &[f64], quantile: f64) -> f64{

        assert_eq!(
            predictions.len(),
            targets.len(),
            "predictions size != targets size (len)"
        );

        assert!(
            quantile > 0.0 && quantile < 1.0,
            "quantile must be between 0 an 1"
        );

        predictions
            .iter()
            .zip(targets.iter())
            .map(|(&p, &t)| {
                let error = t - p;
                if error >= 0.0{
                    quantile*error
                } else {
                    (1.0 - quantile) * -error
                }
            })
            .sum::<f64>()
            /predictions.len() as f64

    }

    //kullback-leibler divergence
    fn kl_divergence(predictions: &[f64], targets: &[f64]) -> f64{
        assert_eq!{
            predictions.len(),
            targets.len(),
            " predictions.len() != targets.len()..!"
        }

        predictions
            .iter()
            .zip(targets.iter())
            .map(|(&p, &t)|{
                if t > 0.0 && p > 0.0 {
                    p * (t / p).ln()
                } else {
                    0.0
                }
            })
            .sum()

    }
}


//focal loss
//hinge loss
//categorical hinge loss
//IoU loss
//Dice loss
//Triplet Loss
