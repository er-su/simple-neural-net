use rand::Rng;

pub fn matrix_vector_mult(rows: usize, cols: usize, mat: &Vec<f64>, vec: &Vec<f64>, out_vec: &mut Vec<f64>) {
    for i in 0..rows {
        let mut sum: f64 = 0.0;
        for j in 0..cols {
            sum += mat[i * cols + j] * vec[j];
        }

        out_vec[i] = sum;
    }
}

pub fn vector_vector_add(vec1: &Vec<f64>, vec2: &Vec<f64>, result: &mut Vec<f64>, size: usize) {
    for i in 0..size {
        result[i] = vec1[i] + vec2[i];
    }
}

pub fn vec_add_in_place(vec1: &mut Vec<f64>, vec2: &Vec<f64>, size: usize) {
    for i in 0..size {
        vec1[i] = vec1[i] + vec2[i];
    }
}

pub fn print_vector(vec: &Vec<f64>, size: usize) {
    for i in 0..size {
        println!("{}", vec[i]);
    }
}

pub fn init_rand_vector(input: &mut Vec<f64>,  size: usize) {
    for _i in 0..size {
        input.push(rand::thread_rng().gen_range(-0.01..=0.01)); // Range from -2 to 2
    }
    
}

pub fn ReLU(input: &Vec<f64>, output: &mut Vec<f64>, size: usize) {
    for i in 0..size {
        if input[i] < 0.0 {
            output[i] = 0.0;
        }
        else {
            output[i] = input[i];
        }
    }
}

pub fn soft_max(input: &Vec<f64>, output: &mut Vec<f64>, size: usize){
    let mut max_val: f64 = input[0];
    for i in 0..size {
        if input[i] > max_val {
            max_val = input[i];
        }
    }

    let mut exponents: Vec<f64> = Vec::new();
    let mut sum: f64 = 0.0;
    for i in 0..size {
        exponents.push((input[i] - max_val).exp());
        sum += exponents[i];
    }

    for i in 0..size {
        output[i] = exponents[i] / sum;
    }
}

pub fn log_soft_max(input: &Vec<f64>, output: &mut Vec<f64>, size: usize) {
    let mut max_val: f64 = input[0];
    for i in 0..size {
        if input[i] > max_val {
            max_val = input[i];
        }
    }

    let mut sum: f64 = 0.0;
    for i in 0..size {
        sum += (input[i] - max_val).exp();
    }

    for i in 0..size {
        output[i] = input[i] - max_val - sum.ln();
    }
}