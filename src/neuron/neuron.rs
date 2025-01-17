use crate::f_not_linear::activation::{relu, leaky_relu, relun, star_relu, shilu};
use crate::f_not_linear::activation::{sigmoid, tanh, parametric_relu, elu, swish, gelu, softmax};

pub fn simple_neuron(input_vec: Vec<f64>, weights_vec: Vec<f64>, bias: f64, bias_weight: f64, activation_func: &str, actiovation_params: Vec<f64> ) -> f64{
    //O tipo str não é válido como parâmetro de função... deve usar &str

    let mut product_vec = input_vec.clone();

    for i in 0..input_vec.len(){
        product_vec[i] = input_vec[i]*weights_vec[i];
    };

    let big_sigma: f64 = product_vec.iter().sum();


    let linear: f64 = big_sigma + (bias*bias_weight);

    let not_linear: f64 = match activation_func {
        "relu" => relu(linear),
        "leaky_relu" => leaky_relu(linear, actiovation_params[0]),
        "relun" => relun(linear, actiovation_params[0]),
        "star_relu" => star_relu(linear, actiovation_params[0], actiovation_params[1]),
        "shilu" => shilu(linear, actiovation_params[0], actiovation_params[1]),
        "sigmoid" => sigmoid(linear),
        "tanh" => tanh(linear),
        "parametric_relu" => parametric_relu(linear, actiovation_params[0]),
        "elu" => elu(linear, actiovation_params[0]),
        "swish" => swish(linear),
        "gelu" => gelu(linear),
        _ => panic!("Função de ativação inválida: {}", activation_func),
    };

    return not_linear;


}


pub fn softmax_neuron(input_vec: Vec<Vec<f64>>, weights_vec: Vec<Vec<f64>>, bias: Vec<f64>, bias_weight: Vec<f64> ) -> Vec<f64>{

    let mut big_sigma_vec: Vec<f64> = Vec::new();

    for v in 0..input_vec.len(){
        let temp_input = &input_vec[v];
        let temp_weights = &weights_vec[v];
        let mut temp_product_vec: Vec<f64> = Vec::new();

        for i in 0..temp_input.len(){
            temp_product_vec.push(temp_input[i] * temp_weights[i]);
        };

        big_sigma_vec.push(temp_product_vec.iter().sum());
    };

    let mut linear_vec = bias.clone();

    for j in 0..big_sigma_vec.len(){
        linear_vec[j] = big_sigma_vec[j] + (bias[j]*bias_weight[j]);

    };

    return softmax(linear_vec);


}