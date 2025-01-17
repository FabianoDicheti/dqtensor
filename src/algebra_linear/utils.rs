use crate::algebra_linear::matrizes::CriarMatriz;

pub fn imprimir_matriz(matriz: &CriarMatriz){
    println!("{}", String::from("- ").repeat(matriz.dados[0].len()*3));
    matriz.dados.iter().for_each(|l| println!(" {:?}", l));
    println!("{}", String::from("- ").repeat(matriz.dados[0].len()*3));

}


pub fn copiar_matriz(matriz: &CriarMatriz) -> CriarMatriz{
    let mut dados: Vec<Vec<f64>> = Vec::new();
    for linha in matriz.dados.clone(){
        dados.push(linha.to_vec());
    }
    return CriarMatriz{linhas:  matriz.linhas, colunas: matriz.colunas, dados};
}