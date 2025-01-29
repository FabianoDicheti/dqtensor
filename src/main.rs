mod algebra_linear;
mod f_not_linear;
mod neuron;
mod optimizers;
mod loss_functions;

use crate::algebra_linear::matrizes::CriarMatriz;
use crate::algebra_linear::utils::{imprimir_matriz, copiar_matriz};
use crate::algebra_linear::operacoes::{aplicar, identidade, somar_matrizes, subtrair_matrizes, multiplicar_matrizes, dividir_itens, multiplicar_itens, resolver_sistema, determinante, matriz_transposta, matriz_inversa};
use crate::f_not_linear::activation::{relu, leaky_relu, relun, star_relu, shilu};
use crate::f_not_linear::activation::{sigmoid, tanh, parametric_relu, elu, swish, gelu, softmax};
use crate::neuron::neuron::{simple_neuron, softmax_neuron};
use crate::optimizers::bp_optimizers::Adam;
use crate::loss_functions::loss_functions::LossFunction;

fn main() {
   
    let matriz_0: CriarMatriz = CriarMatriz::nova_matriz(4, 4);
    println!("{:?}", matriz_0);

    let matriz_1: CriarMatriz = CriarMatriz::ler_txt("src/teste.txt");
    println!("{:?}", matriz_1);

    let matriz_2: CriarMatriz = CriarMatriz::matriz_de_string("5 5 5; 6 6 6; 7 8 9");
    println!("CriarMatriz de string \n{:?}", matriz_2);

    imprimir_matriz(&matriz_2);

    let mut matriz_3: CriarMatriz = copiar_matriz(&matriz_0);
    println!("-- cópia --");
    imprimir_matriz(&matriz_3);

    matriz_3 = aplicar(matriz_3, |x| x+2.0);

    imprimir_matriz(&matriz_3);

    matriz_3 = aplicar(matriz_3, |x| x/7.0);
    println!(" / 7 -----");
    imprimir_matriz(&matriz_3);

    let matriz_1_transposta = matriz_transposta(&matriz_1);
    println!(" matriz 1");
    imprimir_matriz(&matriz_1);
    println!(" transposta de 1 ");
    imprimir_matriz(&matriz_1_transposta);

    let matriz_2_transposta = matriz_transposta(&matriz_2);
    println!(" matriz 2");
    imprimir_matriz(&matriz_2);
    println!(" transposta de 2 ");
    imprimir_matriz(&matriz_2_transposta);


    //criar matriz identidade

    let mut matriz_identidade = CriarMatriz::nova_matriz(5,5);
    println!("matriz inicial");
    imprimir_matriz(&matriz_identidade);

    matriz_identidade = identidade(matriz_identidade);
    println!("matriz identidade");
    imprimir_matriz(&matriz_identidade);

    let matriz_somada = somar_matrizes(&matriz_1, &matriz_2);

    let matriz_subtraida = subtrair_matrizes(&matriz_1, &matriz_2);

    let itens_multiplicados = multiplicar_itens(&matriz_1, &matriz_2);

    let matriz_dividida = dividir_itens(&matriz_1, &matriz_2);

    let matriz_multiplicada = multiplicar_matrizes(&matriz_1, &matriz_2);

    let matriz_sistema = CriarMatriz::matriz_de_string("1 2 3 6 ; 0 1 2 4 ; 0 0 10 30");

    let matriz_solver = resolver_sistema(&matriz_sistema);

    let determinante_da_matriz: f64 = determinante(&matriz_2); 

    let matriz_det1 = CriarMatriz::matriz_de_string("8 5 ; 4 5");
    let det_m1: f64 = determinante(&matriz_det1);

    let matriz_det2 = CriarMatriz::matriz_de_string("1 -2 3 ; 2 0 3 ; 1 5 4");
    let det_m2: f64 = determinante(&matriz_det2);

    let matriz_det3 = CriarMatriz::matriz_de_string("2 4 1 ; 3 4 1 ; 5 1 3");
    let det_m3: f64 = determinante(&matriz_det3);

    println!("matriz 1");
    imprimir_matriz(&matriz_1);
    println!("\nmatriz 2");
    imprimir_matriz(&matriz_2);

    println!("\nsoma das matrizes 1 e 2");
    imprimir_matriz(&matriz_somada);

    println!("\nsubtração das matrizes 1 e 2");
    imprimir_matriz(&matriz_subtraida);

    println!("\nmultiplicacao dos itens da matrize 1 pelos itens da matriz 2");
    imprimir_matriz(&itens_multiplicados);

    println!("\ndivisao dos itens da matrize 1 pelos itens da matriz 2");
    imprimir_matriz(&matriz_dividida);

    println!("\n multiplicação das matrizes 1 e 2");
    imprimir_matriz(&matriz_multiplicada);

    println!("\n resolver sistema matriz 1");
    imprimir_matriz(&matriz_solver);

    println!(" \n calculo de da determinante da matriz 2");
    imprimir_matriz(&matriz_2);

    println!(" ---- determinante igual a => {} ", determinante_da_matriz);

    println!("\ndeterminante da ex 1 = {}, \ndeterminante da ex 2 = {} \ndeterminante da ex 2 = {}", det_m1, det_m2, det_m3);


    let matriz_1_inversa = matriz_inversa(&matriz_det1);
    println!(" matriz 1");
    imprimir_matriz(&matriz_det1);
    println!(" inversa de 1 ");
    imprimir_matriz(&matriz_1_inversa);

    let matriz_2_inversa = matriz_inversa(&matriz_det2);
    println!(" matriz 2");
    imprimir_matriz(&matriz_det2);
    println!(" inversa de 2 ");
    imprimir_matriz(&matriz_2_inversa);

    
    let matriz_3_inversa = matriz_inversa(&matriz_det3);
    println!(" matriz 3");
    imprimir_matriz(&matriz_det3);
    println!(" inversa de 3 ");
    imprimir_matriz(&matriz_3_inversa);

    let matriz_inv1: CriarMatriz = CriarMatriz::matriz_de_string("2 -1 3 ; -5 3 1 ; -3 2 3");
    let matriz_4_inversa = matriz_inversa(&matriz_inv1);
    println!("\n\n teste \n\n teste 1");
    imprimir_matriz(&matriz_inv1);
    println!(" inversa => ");
    imprimir_matriz(&matriz_4_inversa);

    let matriz_inv2: CriarMatriz = CriarMatriz::matriz_de_string("1 2 3 ; 0 1 5 ; 5 6 0");
    let matriz_5_inversa = matriz_inversa(&matriz_inv2);
    println!("\n\n teste \n\n teste 2");
    imprimir_matriz(&matriz_inv2);
    println!(" inversa => ");
    imprimir_matriz(&matriz_5_inversa);


    let x = -2.4;
    let a = 0.2;
    let alpha = 1.0;
    let i_vector = vec![1.0, 2.0, 3.0, 4.0];

    println!("ReLu: {}", relu(x));
    println!("LeakyReLu: {}", leaky_relu(x, 0.1));
    println!("ReLuN: {}", relun(x, 0.1));
    println!("starReLu: {}", star_relu(x, 0.2, 1.0));
    println!("shilLu: {}", shilu(x, 0.2, -1.0));

    println!(" \n sigmoid({}): {}", x, sigmoid(x));
    println!(" tanh({}) : {}",x, tanh(x));
    println!(" param relu ({}, {}): {}",x, a, parametric_relu(x, a));
    println!(" elu ({}, {}) {}", x, alpha, elu(x, alpha));
    println!(" swish({}): {}", x, swish(x));
    println!("gelu ({}): {}", x, gelu(x));
    println!(" softmax({:?}) : {:?}", i_vector.clone(), softmax(i_vector));




    let input_vec = vec![1.0, 2.0, 3.0];
    let weights_vec = vec![0.5, 0.5, 0.5];
    let bias = 1.0;
    let bias_weight = 0.5;
    let activation_func = "sigmoid";
    let activation_params = vec![];

    let result = simple_neuron(input_vec, weights_vec, bias, bias_weight, activation_func, activation_params);
    println!("\n\n neuron \n Resultado: {}", result);

    ////////
    let input_vec = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let weights_vec = vec![vec![0.5, 0.5], vec![0.5, 0.5]];
    let bias = vec![1.0, 1.0];
    let bias_weight = vec![0.5, 0.5];

    let result = softmax_neuron(input_vec, weights_vec, bias, bias_weight);
    println!("\n\n {:?}", result);



    //////////////////////////// L O S S  F U N C T I O N S \\\\\\\\\\\\\\\\\\\\\\\\

    println!("\n\n L O S S  F U N C T I O N S");

    let predictions = vec![0.1, 0.5, 0.7, 0.5];
    let targets = vec![1.0, 0.0, 1.0, 0.0];
    
    let loss_mse = LossFunction::MeanSquaredError;
    println!("\n mse: {}", loss_mse.calculate(&predictions, &targets));

    let loss_crse = LossFunction::CrossEntropy;
    println!("\n cross entropy: {}", loss_crse.calculate(&predictions, &targets));

    let loss_mae = LossFunction::MeanAbsoluteError;
    println!("\n mae: {}", loss_mae.calculate(&predictions, &targets));

    let hub_targets = vec![0.4, 2.0, 8.0, 8.0];
    let loss_huber = LossFunction::HuberLoss(1.0); // o delta nesse caso é 1
    let example_huber = loss_huber.calculate(&predictions, &hub_targets);
    println!("\n huber loss: {}", example_huber);

    let loss_log_cosh = LossFunction::LogCoshLoss;
    let example_log_cosh = loss_log_cosh.calculate(&predictions, &hub_targets);
    println!("\n log cosh loss: {}", example_log_cosh);


    let quantile_targets = vec![0.3, 2.0, 1.0, 8.0];
    let loss_quantile = LossFunction::QuantileLoss(0.75); 
    let example_quantile = loss_quantile.calculate(&predictions, &quantile_targets);
    println!("\n quantile loss: {}", example_quantile);

    let loss_kl = LossFunction::KLDivergence;
    println!("\n kullback-leibler divergence {}", loss_kl.calculate(&predictions, &targets));


    let focal = LossFunction::FocalLoss(0.3, 1.6);
    println!(" \n focall loss {} \n", focal.calculate(&predictions, &targets));

    let hinge = LossFunction::HingeLoss;
    println!("\n hinge loss {} \n", hinge.calculate(&predictions, &targets));


    let preds_hinge = vec![vec![0.1, 0.2, 0.3, 0.4], vec![0.1, 0.8, 0.7, 0.6],vec![1.0, 0.5, 0.75, 0.54]];
    let targets_hinge = vec![vec![0.0, 0.0, 1.0, 0.0], vec![1.0, 1.0, 1.0, 0.0],vec![1.0, 0.0, 0.0, 1.0]];
    let categorical_hinge = LossFunction::CategroricalHingeLoss(preds_hinge.clone(), targets_hinge.clone());
    println!("\nCategorical Hinge Loss: {}\n", categorical_hinge.calculate(&[], &[]));

    let dice = LossFunction::DiceLoss(1e-6);
    println!(" \n Dice loss {} \n", dice.calculate(&predictions, &targets));

    let iou_targ = vec![2.0, 2.0, 4.0, 4.0];
    let iou_pred = vec![1.0, 1.0, 3.0, 4.0];
    let iou_losss = LossFunction::IoULoss(iou_pred.clone(), iou_targ.clone());
    println!("\n Inter or Uni Loss {} \n", iou_losss.calculate(&[], &[]));


    let anchor_tr = vec![1.0, 2.0, 3.0, 2.5];
    let negative_tr = vec![1.1, 2.2, 3.3, 3.3];
    let positive_tr = vec![2.2, 3.0, 4.4, 3.4];
    let margin_tr = 0.4;
    let tripletloss = LossFunction::TripletLoss(anchor_tr.clone(), positive_tr.clone(), negative_tr.clone(), margin_tr.clone());
    println!("\n Triplet Loss {} \n", tripletloss.calculate(&[], &[]));



    /////////////////////////// O P T M I Z E R S \\\\\\\\\\\\\\\\\\\\\\\\\\

    println!("\n\n O P T M I Z E R S");


    //  pub fn new(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64, param_size: usize) -> Self {
    let mut adam = Adam::new(0.001, 0.9, 0.999, 1e-8, 3);

    let mut params = vec![1.0, 2.0, 3.0];


    let grads = vec![0.1, 0.2, -0.1]; // calculated by loss function
    
    println!("\n Adam test ");

    for _ in 0..10{
        adam.update(&mut params, &grads);
        println!("\n Parametros atualizados {:?}", params);
    }

}
