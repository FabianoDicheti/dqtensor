mod algebra_linear;
mod f_not_linear;
mod neuron;

use crate::algebra_linear::matrizes::CriarMatriz;
use crate::algebra_linear::utils::{imprimir_matriz, copiar_matriz};
use crate::algebra_linear::operacoes::{aplicar, identidade, somar_matrizes, subtrair_matrizes, multiplicar_matrizes, dividir_itens, multiplicar_itens, resolver_sistema, determinante, matriz_transposta, matriz_inversa};
use crate::f_not_linear::activation::{relu, leaky_relu, relun, star_relu, shilu};
use crate::f_not_linear::activation::{sigmoid, tanh, parametric_relu, elu, swish, gelu, softmax};
use crate::neuron::neuron::{simple_neuron, softmax_neuron};


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

}
