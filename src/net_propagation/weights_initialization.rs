fn calcular_matriz_inversa_distancias(vetor_entrada: Vec<f64>) -> Vec<Vec<f64>> {
    let n = vetor_entrada.len();
    let valor_inicial = (n + 1) as f64 * 0.1; // Valor inicial dinâmico: (n + 1) * 0.1

    // Inicializa a matriz de saída com zeros
    let mut matriz_saida = vec![vec![0.0; n]; n];

    // Preenche a matriz com os valores do inverso da distância
    for i in 0..n {
        for j in 0..n {
            // Calcula a distância entre os itens i e j
            let distancia = (vetor_entrada[i] - vetor_entrada[j]).abs();

            // Calcula o inverso da distância
            let inverso_distancia = valor_inicial - distancia;

            // Armazena o valor na matriz
            matriz_saida[i][j] = inverso_distancia;
        }
    }

    // Normaliza a matriz para que os valores fiquem entre 0 e 1
    normalizar_matriz(&mut matriz_saida);

    matriz_saida
}

fn normalizar_matriz(matriz: &mut Vec<Vec<f64>>) {
    let n = matriz.len();
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;

    // Encontra o valor mínimo e máximo na matriz
    for i in 0..n {
        for j in 0..n {
            if matriz[i][j] < min_val {
                min_val = matriz[i][j];
            }
            if matriz[i][j] > max_val {
                max_val = matriz[i][j];
            }
        }
    }

    // Normaliza os valores da matriz para o intervalo [0, 1]
    for i in 0..n {
        for j in 0..n {
            matriz[i][j] = (matriz[i][j] - min_val) / (max_val - min_val);
        }
    }
}



use rand::Rng;

fn gerar_matriz_aleatoria(n: usize) -> Vec<Vec<f64>> {
    let mut rng = rand::thread_rng(); // Inicializa o gerador de números aleatórios
    let mut matriz = vec![vec![0.0; n]; n]; // Cria uma matriz n x n preenchida com zeros

    // Preenche a matriz com valores aleatórios entre 0 e 1
    for i in 0..n {
        for j in 0..n {
            matriz[i][j] = rng.gen::<f64>(); // Gera um número aleatório entre 0 e 1
        }
    }

    matriz
}

fn main() {
    let n = 3; // Tamanho do vetor (e da matriz)
    let matriz_aleatoria = gerar_matriz_aleatoria(n);

    // Exibe a matriz gerada
    for linha in matriz_aleatoria {
        println!("{:?}", linha);
    }
}




fn main() {
    let vetor_entrada = vec![1.0, 2.0, 3.0];
    let matriz_saida = calcular_matriz_inversa_distancias(vetor_entrada);

    // Exibe a matriz de saída
    for linha in matriz_saida {
        println!("{:?}", linha);
    }
}