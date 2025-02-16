use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use csv::Reader;
use rand::Rng;


pub struct DataFrame {
    pub columns: Vec<Vec<String>>,
    pub df_cols: Vec<String>,
}

impl DataFrame {
    pub fn from_file(path: &str) -> Result<Self, Box<dyn Error>> {
        let file = File::open(path)?;
        let mut rdr = Reader::from_reader(file);
        
        // Ler cabeçalhos
        let headers = rdr.headers()?.clone();
        let df_cols = headers.iter().map(|h| h.to_string()).collect();
        
        // Inicializar colunas
        let num_columns = headers.len();
        let mut columns = vec![Vec::new(); num_columns];
        
        // Processar registros
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

    pub fn encode_column(data: &[String]) -> Vec<Vec<f32>> {
        let mut unique_values = Vec::new();
        let mut seen = HashMap::new();
    
        // Criar mapeamento de valores únicos com índices
        for value in data {
            if !seen.contains_key(value) {
                let index = unique_values.len(); // Índice baseado na ordem de aparecimento
                seen.insert(value.clone(), index);
                unique_values.push(value.clone());
            }
        }
    
        let num_categories = unique_values.len();
        
        // Aplicar codificação one-hot
        data.iter()
            .map(|value| {
                let &index = seen.get(value).unwrap();
                let mut one_hot = vec![0.0; num_categories];
                one_hot[index] = 1.0;
                one_hot
            })
            .collect()
    }

    pub fn shuffle_rows(&mut self, iterations: i32) {
        for _ in 0..iterations{
            let n = self.columns[0].len();
            
            // Verifica consistência das colunas
            for col in &self.columns {
                assert_eq!(col.len(), n, "Todas as colunas devem ter o mesmo número de linhas");
            }
            
            let comprimento_aleatorio = n / 3;
            let mut rng = rand::thread_rng();
            
            // Cria vetor de índices originais
            let mut indices: Vec<usize> = (0..n).collect();
            
            // Embaralha os índices
            for _ in 0..comprimento_aleatorio {
                let origem = rng.gen_range(0..indices.len());
                let destino = rng.gen_range(0..indices.len());
                
                let valor = indices.remove(origem);
                indices.insert(destino, valor);
            }
            
            // Reorganiza todas as colunas de acordo com os índices embaralhados
            for col in &mut self.columns {
                let mut nova_coluna = Vec::with_capacity(n);
                for &indice in &indices {
                    nova_coluna.push(col[indice].clone());
                }
                *col = nova_coluna;
            }
        }
    }

    pub fn show_head(&self, quant: usize) {
        // Verifica se há colunas
        if self.df_cols.is_empty() {
            println!("DataFrame vazio");
            return;
        }
        
        // Número de linhas para mostrar (máximo 5)
        let total_linhas = self.columns[0].len();
        let mostrar_linhas = std::cmp::min(quant, total_linhas);
        
        // Imprime cabeçalhos
        println!("{}", self.df_cols.join("\t"));
        
        // Imprime linhas
        for i in 0..mostrar_linhas {
            let linha: Vec<&str> = self.columns.iter()
                .map(|coluna| coluna[i].as_str())
                .collect();
            println!("{}", linha.join("\t"));
        }
        
        // Mostra estatísticas
        println!("\n first {} of {} rows", mostrar_linhas, total_linhas);
    }
}
