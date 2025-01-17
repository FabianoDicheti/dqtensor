use crate::algebra_linear::matrizes::CriarMatriz;

pub fn identidade(mut matriz: CriarMatriz) -> CriarMatriz{
    if matriz.colunas != matriz.linhas{
        panic!("matriz não é quadrada!");
    }
    for l in 0..matriz.linhas{
        matriz.dados[l][l] = 1.0;

    }

    return matriz

}

pub fn aplicar(mut matriz: CriarMatriz, f: impl Fn(f64) -> f64) -> CriarMatriz{
    matriz.dados = matriz.dados.iter()
                                .map(|l|{l.iter()
                                    .map(|x|f(*x))
                                    .collect()})
                                    .collect();
    return matriz
}

pub fn aritimetica_matrizes(matriz_a: &CriarMatriz, matriz_b: &CriarMatriz, operacao: i16) -> CriarMatriz{

    if matriz_a.colunas != matriz_b.colunas || matriz_a.linhas != matriz_b.linhas{
        panic!(" Não foi possível somar as duas matrizes, pois o tamanho é diferente!");
    }

    let operacao_fn: Box<dyn Fn(f64, f64) -> f64> = match operacao{
        1 => Box::new(|x, y| x + y),
        2 => Box::new(|x, y| x - y),
        3 => Box::new(|x, y| x * y),
        4 => Box::new(|x, y|{
            if y == 0.0{
                panic!("encontrada divisão por zero");
            }
         x / y}),
        _ => panic!("operacao {} inválida! 1 adição, 2 subtração, 3 multiplicação, 4 divisão", operacao),
    };

    let mut resultado: CriarMatriz = CriarMatriz::nova_matriz(matriz_a.linhas, matriz_a.colunas);
    for i in 0..matriz_a.linhas{
        for j in 0..matriz_b.colunas{
            resultado.dados[i][j] = operacao_fn(matriz_a.dados[i][j], matriz_b.dados[i][j]);
        }
    }

    return resultado
}

pub fn somar_matrizes(matriz_a: &CriarMatriz, matriz_b: &CriarMatriz) -> CriarMatriz{
    return aritimetica_matrizes(matriz_a, matriz_b, 1)
}

pub fn subtrair_matrizes(matriz_a: &CriarMatriz, matriz_b: &CriarMatriz) -> CriarMatriz{
    return aritimetica_matrizes(matriz_a, matriz_b, 2)
}

pub fn multiplicar_itens(matriz_a: &CriarMatriz, matriz_b: &CriarMatriz) -> CriarMatriz{
    return aritimetica_matrizes(matriz_a, matriz_b, 3)
}

pub fn dividir_itens(matriz_a: &CriarMatriz, matriz_b: &CriarMatriz) -> CriarMatriz{
    return aritimetica_matrizes(matriz_a, matriz_b, 4)
}


pub fn multiplicar_matrizes(matriz_a: &CriarMatriz, matriz_b: &CriarMatriz) -> CriarMatriz{
    if matriz_a.linhas != matriz_b.colunas || matriz_a.colunas != matriz_b.linhas{
        panic!("número de linhas de A deve ser igual ao número de colunas de B, e vice-versa!")
    }

    let mut prod = CriarMatriz::nova_matriz(matriz_a.linhas, matriz_b.colunas);
    for linha in 0..matriz_a.linhas{
        for coluna in 0..matriz_b.colunas{
            let mut soma: f64 = 0.0;
            for item in 0..matriz_b.linhas{
                soma += matriz_a.dados[linha][item] * matriz_b.dados[item][coluna];
            }
            prod.dados[linha][coluna] = soma;
        }

        
    }
    return prod
}

////////////////////////////////////////////////////////////////////////////
pub fn resolver_sistema(matriz: &CriarMatriz) -> CriarMatriz{
    let mut matriz_resolvida: CriarMatriz = CriarMatriz::nova_matriz(matriz.linhas, matriz.colunas);
    matriz_resolvida.dados = matriz.dados.clone();

    if matriz_resolvida.dados[0][0] == 0.0{
        matriz_resolvida = inverte_linhas(matriz_resolvida, 0);
    }
    let mut pivo: usize = 0;
    let linhas = matriz_resolvida.linhas;
    while pivo < linhas{
        for linha in 0..linhas{
            let divisor = matriz_resolvida.dados[pivo][pivo];
            if divisor.abs() < 1e-10 {
                panic!("A matriz é singular e não pode ser resolvida usando este método.");
            }
            let multiplicador = matriz_resolvida.dados[linha][pivo]/divisor;
            
            if linha == pivo{
                matriz_resolvida.dados[pivo] = matriz_resolvida.dados[pivo].iter()
                                                        .map(|valor| valor/divisor)
                                                        .collect();
            }
            else{
                for coluna in 0..matriz_resolvida.colunas{
                    matriz_resolvida.dados[linha][coluna] -= matriz_resolvida.dados[pivo][coluna] * multiplicador;
                }
            }

        }
        pivo += 1;
    }
    return arredondamentos_matriz(matriz_resolvida)
}

pub fn inverte_linhas(mut matriz: CriarMatriz, linha: usize) -> CriarMatriz{
    let mut num_linha: usize = 0;
    for lin in 0..matriz.linhas{
        if matriz.dados[lin][0] > 0.0{
            num_linha = lin;
            break;
        }
    }
    let temp_linha: Vec<f64> = matriz.dados[linha].clone();
    matriz.dados[linha] = matriz.dados[num_linha].clone();
    matriz.dados[num_linha] = temp_linha;

    return matriz
}

pub fn arredondamentos_matriz(mut matriz: CriarMatriz) -> CriarMatriz{
    for linha in 0..matriz.linhas{
        for coluna in 0..matriz.colunas{
            let item: f64 = matriz.dados[linha][coluna];
            if item == -0.0{
                matriz.dados[linha][coluna] = 0.0;
            }

            let arredondado: f64 = item.floor();
            if item - arredondado > 0.9999999{
                matriz.dados[linha][coluna] = item.round();
            }

            if item > 0.0 && item < 0.001{
                matriz.dados[linha][coluna] = 0.0;
            }

            if item <0.0 && item > -0.001{
                matriz.dados[linha][coluna] = 0.0;
            }

        }

    }
    return matriz
}

////////////////////////////////////////////////////////////////////////////


pub fn cofator(matriz: &CriarMatriz, linha_evidencia: usize, item: usize) -> f64{
    let mut corte: Vec<Vec<f64>> = Vec::new();

    for linha in 0..matriz.linhas{
        if linha == linha_evidencia{
            continue;
        }
        let mut vetor: Vec<f64> = Vec::new();
        for coluna in 0..matriz.colunas{
            if coluna == item{
                continue;
            }
            vetor.push(matriz.dados[linha][coluna]);


        }
        corte.push(vetor);
    }
    let num_linhas: usize = corte.len();
    let num_colunas: usize = corte[0].len();

    let mut matriz_recursiva: CriarMatriz = CriarMatriz::nova_matriz(num_linhas, num_colunas);
    matriz_recursiva.dados = corte;

    let menor_complementar: f64 = determinante(&matriz_recursiva);
    let base: i32 = -1;

    return menor_complementar * f64::from(base.pow((linha_evidencia + item) as u32));

}

pub fn determinante(matriz: &CriarMatriz) -> f64{
    if matriz.linhas != matriz.colunas{
        panic!("matriz precisa ser quadrada para se calcular a determinante, encontrado uma matriz de {} por {}", matriz.linhas, matriz.colunas);
    }

    if matriz.linhas == 2{
        return matriz.dados[0][0]*matriz.dados[1][1] - matriz.dados[0][1]* matriz.dados[1][0];
    }
    else{
        let linha: usize = 1;
        let mut det: f64 = 0.0;

        for item in 0..matriz.dados[linha].len(){
            det += cofator(matriz, linha, item) * matriz.dados[linha][item];
        }

        return det

    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////

pub fn matriz_transposta(matriz: &CriarMatriz) -> CriarMatriz{
    let mut transposta: CriarMatriz = CriarMatriz::nova_matriz(matriz.colunas, matriz.linhas);
    for linha in 0..matriz.linhas{
        for coluna in 0..matriz.colunas{
            transposta.dados[coluna][linha] = matriz.dados[linha][coluna];
        }
    }
    
    return transposta
}

pub fn matriz_inversa(matriz: &CriarMatriz) -> CriarMatriz{

    let mut matriz_temp: CriarMatriz = CriarMatriz::nova_matriz(matriz.linhas, matriz.colunas);
    matriz_temp.dados = matriz.dados.clone();

    let determinante_temp = determinante(matriz);

    if determinante_temp == 0.0{
        panic!("determinante = 0, não há matriz inversa!")
    }

    if matriz_temp.linhas == 2 || matriz_temp.colunas ==2{
        matriz_temp.dados[0][0] = matriz.dados[1][1]/determinante_temp;
        matriz_temp.dados[1][1] = matriz.dados[0][0]/determinante_temp;
        matriz_temp.dados[0][1] = - matriz.dados[0][1]/determinante_temp;
        matriz_temp.dados[1][0] = - matriz.dados[1][0]/determinante_temp;

        return matriz_temp;
    }

    for linha in 0..matriz_temp.linhas{
        for coluna in 0..matriz_temp.colunas{
            matriz_temp.dados[linha][coluna] = cofator(&matriz, linha, coluna);// usar a matriz original pra calcular, se não... depois de calcular o primeiro, altera o resultado dos demais.
        }
    }

    let mut inversa = arredondamentos_matriz(matriz_temp);

    inversa = matriz_transposta(&inversa);
    inversa = aplicar(inversa, |x| x/determinante_temp);

    return inversa;
}


////////////////////////////////////////////////////////////////////

