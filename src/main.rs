mod f_not_linear;
mod neuron;
mod data;

use neuron::neuron::{Neuron, Layer};
use f_not_linear::activation::ActivationFunction;
use data::ingestion::DataFrame;
use std::error::Error;
use std::env;

fn main() -> Result<(), Box<dyn Error>> {
    // Carregar o CSV
    let mut df = DataFrame::from_file("src/iris.csv")?;

    df.shuffle_rows(42);

    // imprimir o início do dataframe
    df.show_head(10);

    // Pegar os dados da coluna species
    let species_index = df.df_cols.iter().position(|col| col == "species").unwrap() as usize;

    let species_data = &df.columns[species_index];
    
    // Aplicar encoding
    let encoded = DataFrame::encode_column(species_data);
    
    // extrair características
    let features = df.extract_features()?;

    // Mostrar resultados
    println!("Valores únicos na coluna 'species':");
    let unique_values: Vec<&String> = species_data.iter().collect::<std::collections::HashSet<_>>().into_iter().collect();
    println!("{:?}", unique_values);
    
    println!("\nEncoded values:");
    println!("{:?}", encoded);

    println!("\n FEATURES:");
    println!("{:?}", features);


    // neuron forward

    let neuron_test = Neuron::new(ActivationFunction::Sigmoid);

    let layer_test_1 = Layer::new(
        "name_test".to_string(), vec![neuron_test.clone(); 2].into_iter().collect(), 4
    );

    let layer_test_2 = Layer::new(
        "other_name".to_string(), vec![neuron_test.clone(); 3].into_iter().collect(), 2
    );


    let n = features.len();

    for i in 0..n{
        let input_data = &features[i];

        let output_test_1 = layer_test_1.forward(&input_data);

        let output_test_2 = layer_test_2.forward(&output_test_1);

        let softmax_output = ActivationFunction::apply_softmax(&output_test_2);


        println!("\n \n TEST FORWARD OUTPUT {:?} \n", softmax_output);

    }
    

    
    Ok(())
}

    // // Criando diferentes tipos de neurônios
    // let neuron_a = Neuron::new(ActivationFunction::ReLU);
    // let neuron_b = Neuron::new(ActivationFunction::Sigmoid);
    // let neuron_c = Neuron::new(ActivationFunction::Tanh);
    // let neuron_d = Neuron::new(ActivationFunction::LeakyReLU(0.01));

    // // Criando camadas com diferentes tipos de neurônios
    // let layer1 = Layer::new(
    //     "layer_batata".to_string(),
    //     vec![neuron_a.clone(); 2].into_iter()
    //     .chain(vec![neuron_b.clone(); 2])
    //     .chain(vec![neuron_c.clone(); 2])
    //     .chain(vec![neuron_d.clone(); 2]).collect(),
    //     3, // Tamanho da entrada
    // );

    // let saida_layer1 = 48;

    // let layer2 = Layer::new(
    //     "layer_frita".to_string(),
    //     vec![neuron_c.clone(); 3].into_iter()
    //     .chain(vec![neuron_d.clone(); 3])
    //     .chain(vec![neuron_a.clone(); 3])
    //     .chain(vec![neuron_b.clone(); 3])
    //     .collect(),
    //     saida_layer1, // A saída da camada 1 será usada como entrada na camada 2
    // );

    // let saida_layer2 = 12;

    // let layer3 = Layer::new(
    //     "layer_abobora".to_string(),
    //     vec![neuron_a.clone(); 1].into_iter()
    //         .chain(vec![neuron_b.clone(); 1])
    //         .chain(vec![neuron_c.clone(); 1])
    //         .chain(vec![neuron_d.clone(); 1])
    //         .collect(),
    //         saida_layer2, // Tamanho da entrada
    // );

    // // Definindo uma entrada
    // let input_data = vec![1.0, 0.5, 0.2];

    // // Passando os dados pela primeira camada
    // let output_layer1 = layer1.forward(&input_data);
    // println!("\n saida Layer 1: {:?} \n", output_layer1);

    // // Passando a saída da primeira camada como entrada para a segunda
    // let output_layer2 = layer2.forward(&output_layer1);
    // println!(" \n saida Layer 2: {:?} \n", output_layer2);


    // let output_layer3 = layer3.forward(&output_layer2);
    // println!("\n Saida Layer 3: {:?} \n", output_layer3);



