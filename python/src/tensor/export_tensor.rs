use std::fs::File;
use std::io::{self, Write, BufWriter};

fn salvar_matriz_binaria(matriz: &[Vec<f64>], arquivo: &str) -> io::Result<()> {
    let file = File::create(arquivo)?;
    let mut writer = BufWriter::new(file);

    for linha in matriz {
        for &valor in linha {
            writer.write_all(&valor.to_ne_bytes())?; // Salva cada valor como binário
        }
    }

    Ok(())
}

fn main() -> io::Result<()> {
    // Exemplo de matriz de adjacência para uma rede neural densa
    let matriz = vec![
        vec![0.0, 0.5, 0.2],
        vec![0.5, 0.0, 0.8],
        vec![0.2, 0.8, 0.0],
    ];

    salvar_matriz_binaria(&matriz, "rede_neural.bin")?;
    println!("Matriz salva em rede_neural.bin");
    Ok(())
}