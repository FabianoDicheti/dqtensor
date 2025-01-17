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


