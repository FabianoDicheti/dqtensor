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

    pub fn encode_column(data: &[String]) -> Vec<f32> {
        let mut unique_values = Vec::new();
        let mut seen = HashMap::new();
        let mut counter = 1.0;

        // Criar mapeamento de valores únicos
        for value in data {
            if !seen.contains_key(value) {
                seen.insert(value.clone(), counter);
                unique_values.push(value.clone());
                counter += 1.0;
            }
        }

        // Aplicar codificação
        data.iter()
            .map(|value| *seen.get(value).unwrap())
            .collect()
    }


    pub fn shuffle_rows(&mut self) {
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
