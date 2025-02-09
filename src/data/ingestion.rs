use polars::prelude::*;
use polars::io::SerReader; // Importação correta para `CsvReader`
use rand::seq::SliceRandom;
use std::fs::File;
use std::io::BufReader;

/// Função para carregar um CSV, normalizar e dividir em treino, validação e teste
pub fn load_and_split_dataset(
    file_path: &str,
    train_ratio: f64,
    val_ratio: f64,
    test_ratio: f64,
) -> PolarsResult<(DataFrame, DataFrame, DataFrame)> {
    assert!(
        (train_ratio + val_ratio + test_ratio - 1.0).abs() < f64::EPSILON,
        "As proporções devem somar 1.0"
    );

    // Lendo o CSV
    let df = CsvReader::from_path(file_path)?
        .has_header(true)
        .infer_schema(None)
        .finish()?;

    // Convertendo DataFrame para um vetor de linhas para embaralhar
    let mut df_rows = df.clone();
    let mut rng = rand::thread_rng();
    df_rows.as_single_chunk_par()?;
    df_rows.shuffle(&mut rng)?;

    // Divisão dos conjuntos
    let total_rows = df_rows.height();
    let train_size = (total_rows as f64 * train_ratio) as usize;
    let val_size = (total_rows as f64 * val_ratio) as usize;

    let df_train = df_rows.slice(0, train_size);
    let df_val = df_rows.slice(train_size, val_size);
    let df_test = df_rows.slice(train_size + val_size, total_rows - (train_size + val_size));

    // Normalização (min-max scaling)
    let df_train = normalize_dataframe(&df_train)?;
    let df_val = normalize_dataframe(&df_val)?;
    let df_test = normalize_dataframe(&df_test)?;

    // Encoding da coluna de classe
    let df_train = encode_dataframe(&df_train)?;
    let df_val = encode_dataframe(&df_val)?;
    let df_test = encode_dataframe(&df_test)?;

    Ok((df_train, df_val, df_test))
}

/// Normaliza todas as colunas numéricas para [0,1]
fn normalize_dataframe(df: &DataFrame) -> PolarsResult<DataFrame> {
    let mut df_normalized = df.clone();

    for col_name in df.get_column_names() {
        if let Ok(col) = df.column(col_name) {
            match col.dtype() {
                DataType::Float64 | DataType::Int64 | DataType::UInt64 | DataType::Float32 => {
                    let col_f64 = col.cast(&DataType::Float64)?;
                    let min = col_f64.min::<f64>().unwrap_or(0.0);
                    let max = col_f64.max::<f64>().unwrap_or(1.0);
                    if min != max {
                        let norm_col = col_f64.f64()?.apply(|x| (x - min) / (max - min));
                        df_normalized.replace(col_name, norm_col.into_series())?;
                    }
                }
                _ => {} // Ignora colunas não numéricas
            }
        }
    }
    Ok(df_normalized)
}

/// Aplica one-hot encoding na coluna de classe
fn encode_dataframe(df: &DataFrame) -> PolarsResult<DataFrame> {
    let class_col = "species"; // Ajuste conforme necessário
    if let Ok(col) = df.column(class_col) {
        if col.dtype() == &DataType::Utf8 {
            let df_encoded = df.get_dummies(Some(&[class_col.to_string()]))?;
            return Ok(df_encoded);
        }
    }
    Ok(df.clone())
}
