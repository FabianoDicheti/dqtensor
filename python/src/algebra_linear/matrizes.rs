use std::fs;
#[derive(Debug)]

pub struct CriarMatriz{
    pub linhas: usize,
    pub colunas: usize,
    pub dados: Vec<Vec<f64>>,
}

impl CriarMatriz{
    pub fn nova_matriz(linhas: usize, colunas: usize) -> CriarMatriz{
        let dados: Vec<Vec<f64>> = vec![vec![0.0; colunas]; linhas];

        return CriarMatriz{ linhas, colunas, dados};
    }

    pub fn ler_txt(path: &str) -> CriarMatriz{
        let conteudo: String = fs::read_to_string(path).unwrap_or_else(|e| panic!("{e}"));

        let mut matriz: Vec<Vec<f64>> = Vec::new();
        for linhas in conteudo.lines(){
            let mut linha: Vec<f64> = Vec::new();
            let leitura: Vec<&str> = linhas
                                        .split_whitespace()
                                        .collect();
            for ler in leitura{
                linha.push(ler.parse::<f64>().unwrap());
            }
            matriz.push(linha);

        }
        let l: usize = matriz.len();
        let c: usize = matriz[0].len();

        return CriarMatriz{linhas: l, colunas: c, dados: matriz};
    }

    pub fn matriz_de_string(entrada: &str) -> CriarMatriz{
        let mut dados: Vec<Vec<f64>> = Vec::new();
        let linhas: Vec<&str> = entrada.split(";").collect();

        for linha in linhas{
            let registros: Vec<&str> = linha.split_whitespace().collect();
            let mut temp_linha: Vec<f64> = Vec::new();
            for reg in registros{
                temp_linha.push(reg.parse::<f64>().unwrap());
            }
            dados.push(temp_linha);
        }
        let n_l= dados.len();
        let n_c= dados[0].len();
        return CriarMatriz{linhas: n_l, colunas: n_c, dados};

    }

}