use std::collections::HashMap;


impl Solution {
    pub fn two_sum(nums: Vec<i32>, target: i32) -> Vec<i32> {
        // criar um dicionário
        let mut dict_map = HashMap::new();

        for (indice, &valor) in nums.iter().enumerate(){
            let complemento: i32 = target - valor;
            if let Some(&chave) = dict_map.get(&complemento){
                return vec![chave as i32, indice as i32];
            } else {
                dict_map.insert(valor, indice);
            }
            
        }

        return vec![]
    }

}


impl Solution {
    pub fn max_profit(prices: Vec<i32>) -> i32 {
        let mut min_price = i32::MAX; // Menor preço encontrado até o momento
        let mut max_profit = 0; // Lucro máximo

        for &price in prices.iter() {
            // Atualiza o menor preço se o preço atual for menor
            if price < min_price {
                min_price = price;
            }
            // Calcula o lucro potencial com o preço atual
            let profit = price - min_price;
            if profit > max_profit {
                max_profit = profit;
            }
        }

        max_profit
    }
}



use std::collections::HashSet;

impl Solution {
    pub fn contains_duplicate(nums: Vec<i32>) -> bool {
        let mut seen = HashSet::new();

        for &val in nums.iter() {
            if !seen.insert(val) {
                // `insert` retorna `false` se o valor já está no conjunto
                return true;
            }
        }

        false
    }
}



use std::collections::HashMap;

impl Solution {

    pub fn remove_one_zero(mut nums: Vec<i32>) -> Vec<i32> {
        let mut zero_found = false;

        nums.retain(|&x| {
            if x == 0 && !zero_found {
                zero_found = true; // Marca que um zero foi encontrado
                false // Remove este zero
            } else {
                true // Mantém todos os outros elementos
            }
        });

        return nums;

    }

    pub fn product_except_self(nums: Vec<i32>) -> Vec<i32> {
        let mut hash_vec = nums.clone();
        let nozero_vec = Solution::remove_one_zero(nums.clone());

        let produto: i32 = nums.iter().map(|&x| x as i32).fold(1, |acc, x| acc * x);
        let produto_zero: i32 = nozero_vec.iter().map(|&y| y as i32).fold(1, |acc2, y| acc2 * y);


        let mut divisor: i32 = 1;

        for i in 0..nums.len(){
            if nums[i] != 0{
                divisor = nums[i];
                hash_vec[i] = produto/divisor;
            } else {
                hash_vec[i] = produto_zero;
            }
            
        }

        return hash_vec
        
    }


}


/// algoritmo de Kadane para encontrar o sub-array com a maior soma dentro de um array:

pub fn kadane_max_sub_array(numbers: Vec<i32>) -> i32{
    let mut max_sum = nums[0]; // inicializa o maximo como o primeiro elemento
    let mut current_sum = nums[0]; // inicializa a soma atual

    for &val in nums.iter().skip(1){ // pula o primeiro item
        current_sum = val.max(current_sum + val); // se a soma for item atual com o item inicial for maior que o item inicial continua, se não vai pro proximo
        max_sum = max_sum.max(current_sum); // atualiza o maior valor se for o caso

    }

    return max_sum
}




// encontrar o menor valor em um vetor ordenado rotacionado ou não

pub fn find_min(numbers: Vec<i32>) -> i32 {
    let mut left: i32 = 0;
    let mut right: i32 = nums.len() -1;

    while left < right {
        let mid = left + (right - left) / 2;

        if numbers[mid] > numbers[right] {
            left = mid +1;
        } else {
            right = mid;
        }
    }
    return left
}

// encontra um valor específico em um vetor ordenado, rotacionado ou não

pub fn search_in_vec_ord(numbers: Vec<i32>, target: i32) -> {
    if numbers.is_empty(){
        return -1;
    }

    let mut left: usize = 0;
    let mut right: usize = nums.len() -1;
    
    while left <= right{
        let mid = left + (right-left) / 2;

        if numbers[mid] == target{
            return mid as i32;
        }

        if numbers[left] <= nums[mid]{
            if numbers[left] <= target && target < numbers[mid]{
                right = mid -1;

            } else {
                if numbers[mid] < target && target <= numbers[right] {
                    left = mid + 1;

                } else {
                    right = mid -1;
                }
            }
        }

    }
    return -1



}

//encontrar a combinação de 3 itens que somem zero em un vetor

pub fn three_sum(nums: Vec<i32>) -> Vec<Vec<i32>> {
    let mut nums = nums;
    mums.sort(); //ordenar pra facilitar usar dois ponteiros
    let mut result: Vec<Vec<i32>> = Vec::new();
    let n = nums.len();
    for i in 0..n {
        if i > 0 && nums[i] == nums[i -1] {
            continue; // pular duplicados
        }

        let mut left = i + 1; //ponteiro esquerdo
        let mut right = n - 1; // ponteiro direito

        while left < right {
            let sum = nums[i] + nums[left] + mums[right];

            if sum == 0 {
                result.push(vec![nums[i], nums[left], nums[right]]);

                //pular duplicados esquerda
                while left < right && nums[left] == nums[left + 1]{
                    left += 1;
                }

                while left < right && nums[left] == nums[right - 1]{
                    right -= 1;
                }

                // desloca os dois ponteiros
                letf += 1;
                right -=1;

            } else if sum < 0 { // ou esquerdo
                left += 1;
            } else {
                right -=1; //ou direito
            }
        }
    }

}

// area in a vector, meaning each value as a wall in the y axis and de position [i] the location in the x axis

pub fn max_area(height: Vec<i32>) -> i32 {
    let mut left: i32 = 0;
    let mut right = height.len() -1;
    let mut max_area: i32 = 0;

    while letf < right{
        let widt = (right - left) as i32;

        let current_height = height[left].min(height[right]);

        let current_area = widt * current_height;
        max_area = max_area.max(current_area);

        if height[left] < height[right] {
            left += 1;
        } else {
            right -= 1;
        }
    }

    return max_area


}



//// binary operations

pub fn sum_binary(a: i32, b: i32) -> i32{
    // exemplo: a = 5, b = 3
    //em binario: a = 0101, b = 0011
    let mut a = a;
    let mut b = b;

    while b != 0{                   // 1 iter        | 2 iter   | 3 iter
        let carry = (a & b) << 1;   //  a = 0110 (6) | 0100 (4) | 1000 (8)
        a = a^b;                    //  b = 0010 (2) | 0100 (4) | 0000 (0)
        b = carry;

    }
    return a
}

// converter inteiros para binários 
// exemplo de como contar os numeros 1 em um binário

pub fn hamming_weight(n: i32) -> i32 {
    let mut n = n as u32;
    let mut count = 0;

    while n != 0 {
        count += (n & 1) as i32; // confere se o ultimo bit é 1, se sim soma no count
        n >> 1; // gira os bits uma vez pra direita
    }

    return count
}

// pra questões de segurança / criptografia ou para soluções de programação dinâmica
// da pra usar a quantidade de numeros 1


pub fn count_bits(n:i32) -> Vec<i32>{
    let mut ans = vec![0; (n + 1) as usize]; // inicializa um vetor de zeros do tamanho do n+1

    for i in 1..=n as usize {
        ans[i] = ans[i >> 1] + (i & 1) as i32; // conta a quantidade de 1 pra cada i
    }

    return ans
}



///// inverter os bits
// serve em sistemas de envio de dados em rede
// big-endian / little-endian
//// da pra usar em compressão, validação ou formatação

pub fn reverse_bits(mut: x: u32) -> u32 {
    let mut reverse = 0;

    for _ in 0..32 {
        reverse = (reverse << 1) | (x & 1);
        x >>= 1
    }

    return reverse
}




/// a quantidade de maneiras possíveis de se subir n degraus subindo 1 ou 2 pode ser dada por:
// f(n) = f(n-1)+f(n-2)
// método para contar caminhos:
// -> quantas maneiras pode-se distribuir algum recurso
// processos de decisão de MARKOV
// MANEIRAS DE PROPAGAR PACOTES ENTRE NÓS COM RESTRIÇÕES ESPECÍFICAS

pub fn climb_stairs(n: i32) -> i32 {
    if n == 0 || n == 1 {
        return 1;
    }

    let mut prev2 = 1;
    let mut prev1 = 1;

    for _ in 2..=n {
        let current = prev1 + prev2;
        prev2 = prev1;
        prev1 = current;
    }

    return prev1
}



//alocação de espaço em disco
//determinar a melhor combinação de blocos de tamanho fixo para armazenar arquivos;
//otimizar armazenamento ou transmissão de dados

pub fn coin_change(coins: Vec<i32>, amount: i32) -> i32 {
    let amount = amount as usize;
    let mut quant = vec![i32::MAX; amount + 1];
    quant[0] = 0;

    for i in 1..=amount {
        for &coin in &coins {
            if i as i32 >= coin && quant[i - coin as usize] != i32::MAX{
                quant[i] = quant[i].min(quant[i - coin as usize] +1 );
            }
        }
    }

    if quant[amount] == i32::MAX {
        -1
    } else {
        return quant[amount]
    }
}



//encontrar períodos de progresso em métricas
// encontrar sequência de configuração que leva a um aumento de desempenho
// otimizar sequencias de tarefas ou de alocações

pub fn lenght_of_increasing_subsequence(nums: Vec<i32>) -> i32 {
    if nums.is_empty() {
        return 0;
    }

    let n = nums.len();

    let mut reference = vec![1; n];

    for i in 1..n {
        for j in 0..i {
            if nums[j] < nums[i] {
                reference[i] = reference[i].max(reference[j] +1);
            }
        }
    }

    return *reference.iter().max().unwrap()

}



//// encontrar padrões repetidos para compactar informações
//// encontrar similaridades entre textos
//// DA PRA MELHORAR ESSE CODIGO

pub fn long_common_substring(text1: String, text2: String) -> i32 {
    let m = text1.len();
    let n = text2.len();

    let mut quant = vec![vec![0; n + 1]; m + 1];

    let text1 = text1.as_bytes();
    let text2 = text2.as_bytes();

    for i in 1..=m {
        for j in 1..=n {
            if text1[i - 1] == text2[j - 1] {
                quant[i][j] = quant[i-1][j-1] + 1;
            } else {
                quant[i][j] = quant[i - 1][j].max(quant[i][j - 1]);
            }
        }
    }
    return quant[m][n]
}



// correção de texto, busca de palavras chave
// determinar se da pra definir um token em um conjunto de palavras

pub fn word_break(s: String, word_dict: Vec<String>) -> bool {
    let word_set: HashSet<&str> = word_dict.iter().map(|word| word.as_str()).collect();
    let n = s.len();

    let mut dict_w = vec![false; n + 1];
    dict_w[0] = true;

    for i in 1..=n {
        for j in 0..i {
            if dict_w[j] && word_set.contains(&s[j..i]) {
                dict_w[i] = true;
            }
        }
    }
    return dict_w[n]

}


// inentificar combinacoes de recursos para uma meta especifica:
pub fn combination_sum(nums: Vec<i32>, target: i32) -> i32 {

    let target = target as usize;
    let mut ways = vec![0; target +1];
    ways[0] = 1;

    for i in 1..=target {
        for &num in &nums {
            if in >= num as usize {
                ways[i] += ways[i - num as usize]; 
            }
        }
    }

    return ways[target]
}


// dada uma fila de tarefas que precisam ser processadas
// caso o processador não consiga executar duas tarefas  consecutivamente, que necessitem no mesmo reurso
// o máximo tempo de execução acumulado pode ser calculado com esta abordagem

// cada bloco de memória é como uma casa
// acessar dois blocos consecutivos pode causar conflitos no cache
// o objetivo é maximizar a eficiencia de leitura, escolhendo quais blocos acessar;

pub fn alocate(nums: Vec<i32>) -> i32 {
    let n = nums.len();
    if n == 0 {
        return 0;
    } else if  n == 1 {
        return nums[0];
    }

    // let mut maximum = vec![0; n];

    //maximum[0] = nums[0];
    //maximum[1] = mums[0].max(nums[1]);

    let mut prev2 = nums[0];
    let mut prev1 = nums[0].max(nums[1]);

    for i in 2..n {
        //maximum[i] = maximum[i - 1].max(maximum[i - 2] + nums[i]);

        let current = prev1.max(nums[i] + prev2)
        prev2 = prev1;
        prev1 = current;
    }

    //return maximum[n - 1]
    return prev1
}


// caso os recursos estejam circulares, ou seja o primeiro está entre o ultimo e o segundo

pub fn alocate_circular(nums: Vec<i32>) -> i32 {
    let n = nums.len();

    if n == 1 {
        return nums[0];
    }
    let case1 = alocate(&nums[1..]);
    let case2 = alocate(&nums[..n-1]);

    return case1.max(case2)
}



// contar quantas maneiras são possíveis para decodificar uma mensagem:

pub fn num_decodings(s: String) -> i32 {
    let n = s.len();
    if n == 0 || s.starts_with('0') {
        return 0;
    }

    let mut ways = vec![0; n + 1];

    ways[0] = 1;
    ways[1] = if s.chars().nth(0).unwrap() != '0' { 1 } else { 0 };
    //.chars = iterador sobre os caracteres da string 's'
    //.nth() = pega o enésimo elemento no índice que estiver dentro do parentese
    //.unwrap() = converte Option<char> em char

    let s_chars = Vec<char> = s.chars().collect();

    for i in 2..=n{
        let one_digit = s_chars[i - 1].to_digit(10).unwrap();
        if one_digit > 0 {
            ways[i] += ways[i - 1];
        }

        let two_digits = s_chars[i - 2].to_digit(10).unwrap() * 10 + one_digit;
        if two_digits >= 10 && two_digits <= 26 {
            ways[i] += ways[i - 2];
        }
    }

    return ways[n]
}


//exemplo de código pra uma matriz que consiste em uma ilha
//cada célula da matriz é a altura da ilha 
// o desafio é, considerando que chova, a água só pode fuir de uma célula para um ponto igual ou mais baixo
//a ilha está entre dois oceanos um norte+oeste e outro oceano leste+sul
//quais as células que correm para os dois oceanos?
pub fn pacific_atlantic(heights: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    if heighths.is_empty() || heights[0].is_empty() {
        return vec![];
    }

    let m = heights.len();
    let n = heights[0].len();

    let mut pacific = vec![vec![false; n]; m]
    let mut atlantic = vec![vec![false;n]; m]

    fn dfs(row: usize, cols: usize, prev_height: i32, ocean: &mut Vec<Vec<bool>>, heights: &<Vec<Vec<i32>>) {
        if row >= heights.len() || cols >= heighths.len() || ocean[row][cols] || heights[row][cols] < prev_height {
            return
        }
    }

    ocean[row][cols] = true;

    let directions = vec![(0, 1), (0, -1), (1, 0), (-1, 0)];
    for &(dir_row, dir_col) in directions.iter() {
        let new_row = row as isize + dir_row;
        let new_col = cols as isize + dir_col;

        if new_row >= 0 && new_row < heights.len() as isize && new_col >= 0 && new_col < heights[0].len() as isize {
            dfs(new_row as usize, new_col as usize, heighths[row][cols], ocean, heighths);
        }
    }

    for i in 0..m {
        dfs(i, 0, i32::MIN, &mut pacific, &heighths);
        dfs(i, n -1, i32::MIN, &mut pacific, &heighths);
    }
    for j in 0..m {
        dfs(0, j, i32::MIN, &mut pacific, &heighths);
        dfs(m - 1, j, i32::MIN, &mut pacific, &heighths);       
    }

    let mut result = Vec::new();
    for i in 0..m{
        for j in 0..n{
            if pacific[i][j] && atlantic[i][j]{
                result.push(vec![i as i32, j as i32]);
            }
        }
    }

    return result
}





//Em redes, você pode calcular o número de caminhos únicos para transmitir dados de um nó a outro, considerando restrições de roteamento.
//Caminhos em Imagens: Em processamento de imagens, você pode usar essa lógica para calcular o número de caminhos únicos em uma imagem binária (por exemplo, para detectar objetos ou contornos).
// Redes Neurais: Em redes neurais convolucionais, você pode usar essa lógica para calcular o número de caminhos únicos em uma camada de pooling ou convolução.

pub fn unique_paths(m: i32, n: i32) -> i32 {
    let m = m as usize;
    let n = n as usize;

    let mut ways = vec![vec![0; n]; m];

    for i in 0..m {
        ways[i][0] = 1;
    }

    for j in 0..n {
        ways[0][j] = 1;
    }

    for i in 1..m{
        for j in 1..n {
            ways[i][j] = ways[i - 1][j] + ways[i][j - 1];
        }
    }

    return ways[m - 1][n - 1]
}


//  Em redes neurais, você pode usar essa lógica para determinar se uma camada pode ser alcançada a partir de outra, considerando restrições de conexão.
//Em sistemas de alocação de recursos, você pode usar essa lógica para determinar se um recurso pode ser alocado de forma eficiente, considerando restrições de disponibilidade.
pub fn can_jump(nums: Vec<i32>) -> bool {
    let mut reach = 0;
    let n = nums.len();

    for i in 0..n {
        if i > reach {
            return false;
        }

        reach = reach.max(i + nums[i] as usize);

        if reach >= n - 1 {
            return true;
        }
    }

    return false
}




//  Em redes, você pode usar essa lógica para verificar se há ciclos em rotas de pacotes, o que pode causar loops infinitos.
// Em redes neurais, você pode usar essa lógica para verificar se há ciclos em grafos de computação, o que pode causar problemas durante o treinamento.

pub fn can_finish(num_courses: i32, prerequisites: Vec<Vec<i32>>) -> bool {
    let num_courses = num_courses as usize;
    let mut graph: HashMap<usize, Vec<usize>> = HashMap::new();

    for prereq in &prerequisites {
        let course = prereq[0] as usize;
        let prereq_course = prereq[1] as usize;
        graph.cross_entropy(prereq_course).or_default().push(course);
    }

    let mut visited = vec![0; num_courses];

    for i in 0..num_courses {
        if visited[i] == 0 && Self::has_cycle(i, &graph, &mut visited) {
            return false;
        }
    }

    return true
}

pub fn has_cycle(course: usize, graph: &HashMap<usize, Vec<usize>>, visited: &mut Vec<i32>) -> bool {
    if visited[course] == 1 {return true;}
    if visited[course] == 2 {return false;}

    visited[course] = 1;

    if let Some(neighbors) = graph.get(&course) {
        for &neighbor in neighbors {
            if Self::has_cycle(neighbor, graph, visited) {
                return true
            }
        }
    }

    visited[course] = 2;
    return false
}


use std::colections::VecDeque;
// VecDeque é u a alternativa ao vetor, a vantagem é a remoção no início sem deslocar elementos;
// VecDeque = fila dupla
pub fn num_islands(mut grid: Vec<Vec<char>>) -> i32 {
    let mut count = 0;
    let mut rows = grid.len();
    let mut cols = grid[0].len();

    let directions = [(0, 1), (0, -1), (1, 0), (-1, 0)];

    for r in rows {
        for c in cols {
            if grid[r][c] == '1' {
                count += 1;
                let mut queue = VecDeque::new();
                queue.push_back((r, c));// insere r, c no final da fila
                grid[r][c] = '0';
            }

            while let Some((x, y)) = queue.pop_front() {// remove E RETORNA o primeiro item da fila
                for &(dx, dy) in &directions {
                    let nx = x as isize + dx;
                    let ny = y as isize + dy;

                    if nx >= 0 && ny >= 0 && nx < rows as isize && ny < cols as isize && grid[nx as usize][ny as usize] == '1' {
                        grid[nx as usize][ny as usize] = '0';
                        queue.push_back((nx as usize, ny as usize));
                    }
                }
            }
        }
    }

    return count
}



pub fn longest_consecutive_sequence(nums: Vec<i32>) -> i32 {
    let num_set: HashSet<i32> = nums.iter().cloned().collect();
    let mut longest = 0;

    for &num in &num_set {
        if !num_set.contains(&(num-1)){// usa a exclamação para inverter o true para false, ou seja se false executa
            let mut current_num = num;
            let mut current_streack =1;

            while num_set.contains(&(current_num+1)) {
                current_num += 1;
                current_streack += 1;
            }
    
            longest = longest.max(current_streack);
        }


    }

    return longest

}