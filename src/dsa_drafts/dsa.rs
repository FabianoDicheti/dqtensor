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