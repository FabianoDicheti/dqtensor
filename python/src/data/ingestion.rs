use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use csv::ReaderBuilder;

pub struct DataFrame {
    pub columns: Vec<Vec<String>>,
    pub df_cols: Vec<String>,
}

impl DataFrame {
    pub fn from_file(path: &str) -> Result<Self, Box<dyn Error>> {
        let delimiter = detect_delimiter(path)?;
        println!("Delimitador do arquivo csv: '{}'", delimiter as char);

        let file = File::open(path)?;
        let mut rdr = ReaderBuilder::new()
            .delimiter(delimiter)
            .from_reader(file);

        let headers = rdr.headers()?.clone();
        let df_cols = headers.iter().map(|h| h.to_string()).collect();

        let num_columns = headers.len();
        let mut columns = vec![Vec::new(); num_columns];

        for result in rdr.records() {
            let record = result?;
            for (i, field) in record.iter().enumerate() {
                if i < num_columns {
                    columns[i].push(field.to_string());
                } else {
                    return Err("CSV com número inconsistente de colunas".into());
                }
            }
        }

        Ok(DataFrame { columns, df_cols })
    }

    pub fn extract_features(&self) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
        let num_rows = self.columns[0].len();
        let num_features = self.columns.len() - 1;

        let mut features = Vec::with_capacity(num_rows);

        for i in 0..num_rows {
            let mut row = Vec::with_capacity(num_features);
            for j in 0..num_features {
                let value = &self.columns[j][i];
                row.push(value.parse::<f64>()?);
            }
            features.push(row);
        }

        Ok(features)
    }

    pub fn show_head(&self, quant: usize) {
        if self.df_cols.is_empty() {
            println!("DataFrame vazio");
            return;
        }

        let total_linhas = self.columns[0].len();
        let mostrar_linhas = std::cmp::min(quant, total_linhas);

        println!("{}", self.df_cols.join("\t"));

        for i in 0..mostrar_linhas {
            let linha: Vec<&str> = self.columns.iter()
                .map(|coluna| coluna[i].as_str())
                .collect();
            println!("{}", linha.join("\t"));
        }

        println!("\n first {} of {} rows", mostrar_linhas, total_linhas);
    }
}


///  CSV usa ',' ou ';'
fn detect_delimiter(path: &str) -> Result<u8, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut first_line = String::new();
    reader.read_line(&mut first_line)?;

    let count_comma = first_line.matches(',').count();
    let count_semicolon = first_line.matches(';').count();

    if count_semicolon > count_comma {
        Ok(b';')
    } else {
        Ok(b',')
    }
}


/// Normalização Min-Max
pub fn normalize_min_max(data: &mut Vec<Vec<f64>>) {
    if data.is_empty() {
        return;
    }

    let num_features = data[0].len();

    for j in 0..num_features {
        let column: Vec<f64> = data.iter().map(|row| row[j]).collect();
        let min = column.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = column.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if (max - min).abs() < 1e-10 {
            continue; // Evitar divisão por zero
        }

        for row in data.iter_mut() {
            row[j] = (row[j] - min) / (max - min);
        }
    }
}
