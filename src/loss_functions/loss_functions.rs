use std::f64;

pub enum LossFunction {
    MeanSquaredError,
    CrossEntropy,
    MeanAbsoluteError,
    HuberLoss(f64),
    LogCoshLoss,
    QuantileLoss(f64),
    KLDivergence,
    FocalLoss(f64, f64),
    HingeLoss,
    CategroricalHingeLoss(Vec<Vec<f64>>, Vec<Vec<f64>>),
}

impl LossFunction {

    pub fn calculate( &self, predictions: &[f64], targets: &[f64]) -> f64 {
        match self {
            LossFunction::MeanSquaredError => Self::mean_squared_error(predictions, targets),
            LossFunction::CrossEntropy => Self::cross_entropy(predictions, targets),
            LossFunction::MeanAbsoluteError => Self::mean_absolute_error(predictions, targets),
            LossFunction::HuberLoss(delta) => Self::huber_loss(predictions, targets, *delta),
            LossFunction::LogCoshLoss => Self::log_cosh_loss(predictions, targets),
            LossFunction::QuantileLoss(quantile) => Self::quantinle_loss(predictions, targets, *quantile),
            LossFunction::KLDivergence => Self::kl_divergence(predictions, targets),
            LossFunction::FocalLoss(alpha, gamma) => Self::focal_loss(predictions, targets, *alpha, *gamma),
            LossFunction::HingeLoss => Self::hinge_loss(predictions, targets),
            LossFunction::CategroricalHingeLoss(vec_predictions, vec_targets) => {
                Self::categorical_hinge_loss(vec_predictions.clone(), vec_targets.clone())
            }
            

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

    fn focal_loss(predictions: &[f64], targets: &[f64], alpha: f64, gamma: f64) -> f64 {
        
        assert_eq!{
            predictions.len(),
            targets.len(),
            "predictions lenght != targets lenght"
        }

        predictions
            .iter()
            .zip(targets.iter())
            .map(|(&p, &t)|{
                let p_t = if t == 1.0 {p} else {1.0 - p};
                if p_t > 0.0 {
                    -alpha * (1.0 - p_t).powf(gamma) * p_t.ln()
                } else {
                    0.0
                }
            })
            .sum::<f64>()
            / predictions.len() as f64


    }


    fn hinge_loss(predictions: &[f64], targets: &[f64]) -> f64{

        assert_eq!{
            predictions.len(),
            targets.len(),
            " predictions lenght =! targets lenght."
        }

        predictions
            .iter()
            .zip(targets.iter())
            .map(|(&p, &t)|{
                let margin = 1.0 - t * p;
                margin.max(0.0)
            })
            .sum::<f64>()
            /predictions.len() as f64

    }

    fn categorical_hinge_loss(vec_predictions: Vec<Vec<f64>>, vec_targets: Vec<Vec<f64>>) -> f64 {
        assert_eq!(
            vec_predictions.len(),
            vec_targets.len(),
            "predictions length != targets length."
        );

        let num_samples = vec_predictions.len();
        let margin = 1.0;
        let mut loss = 0.0;

        for (pred, target) in vec_predictions.iter().zip(vec_targets.iter()) {
            let true_class_index = target.iter().position(|&x| x == 1.0).unwrap();  
            // position(|&x| x == 1.0) encontra o índice onde o valor é 1.0 (indicando a classe correta).
            //.filter(|&(j, _)| j != true_class_index) verifica cada par (j, &score):
            //Mantém os pares onde o índice 𝑗 não é igual ao índice da classe correta (true_class_index).
            //Descarta o par onde j = true_class_index, porque não queremos considerar a pontuação da classe correta neste cálculo.
            //|&(j, _)|: É a assinatura do fechamento usado pelo filter
            //&: Desreferencia os valores recebidos do iterador (necessário porque estamos iterando sobre referências com .iter()).
            //(j, _): Um "desempacotamento" do par (índice, valor) fornecido pelo .enumerate().
            
            let true_class_score = pred[true_class_index];

            let max_wrong_class_score = pred
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != true_class_index) // O método .filter() é usado para remover elementos de um iterador com base em uma condição.
                .map(|(_, &score)| score)
                .fold(f64::NEG_INFINITY, f64::max);

            loss += (max_wrong_class_score - true_class_score + margin).max(0.0);
        }

        loss / num_samples as f64


    }
}



//categorical hinge loss
//IoU loss
//Dice loss
//Triplet Loss
