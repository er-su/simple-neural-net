use util::init_rand_vector;

pub (crate) mod util;
pub struct Layer {
    num_inputs: usize,
    num_outputs: usize,
    weights: Vec<f64>,
    bias: Vec<f64>,
    preoutput: Vec<f64>,
    output: Vec<f64>,
    activation: fn(&Vec<f64>, &mut Vec<f64>, usize),
}

impl Layer {
    pub fn new(inputs: usize, outputs: usize, actfunc: fn(&Vec<f64>, &mut Vec<f64>, usize)) -> Self {
        let mut layer = Layer {
            num_inputs: inputs,
            num_outputs: outputs,
            weights: vec![0.0; inputs * outputs],
            bias: vec![0.0; outputs],
            preoutput: vec![0.0; outputs],
            output: vec![0.0; outputs],
            activation: actfunc,
        };

        layer.init_rand_params();

        layer
    }

    pub fn print_info(&self) {
        println!("Inputs: {0}", self.num_inputs);
        println!("OUtputs: {0}", self.num_outputs);
        println!("Number of Parameters: {0}", self.num_inputs * self.num_outputs + self.num_outputs);
    }

    pub fn forward(&mut self, input: Vec<f64>, size: usize) -> Vec<f64> {
        assert_eq!(size, self.num_inputs);
        util::matrix_vector_mult(self.num_outputs, self.num_inputs, &self.weights, &input, &mut self.preoutput);

        util::vec_add_in_place(&mut self.preoutput, &self.bias, self.num_outputs);

        (self.activation)(&self.preoutput, &mut self.output, self.num_outputs);

        return self.output.clone();
    }

    pub fn calc_layer_error(&self, prev_error: &Vec<f64>, prev_weights: &Vec<f64>, prev_num_inputs: usize, prev_num_outputs: usize) -> Vec<f64> {
        assert_eq!(prev_num_inputs, self.num_outputs);

        let mut activation_der: Vec<f64> = Vec::new();
        for i in 0..self.num_outputs {
            if self.preoutput[i] < 0.0 {
                activation_der.push(0.0);
            }
            else {
                activation_der.push(1.0);
            }
        }

        let mut layer_error: Vec<f64> = vec![0.0; self.num_outputs]; 
        for i in 0..prev_num_inputs {
            let mut sum = 0.0_f64;
            for j in 0..prev_num_outputs {
                sum += prev_weights[j * prev_num_inputs + i] * prev_error[j];
            }
            layer_error[i] = sum;
        }

        for i in 0..self.num_outputs {
            layer_error[i] = layer_error[i] * activation_der[i];
        }

        return layer_error;
        
    }

    pub fn calc_last_error(&self, label: usize) -> Vec<f64> {
        let mut layer_error: Vec<f64> = self.output.clone();
        layer_error[label] = layer_error[label] - 1.0;

        return layer_error;
    }

    pub fn calc_loss(&self, label: usize) -> f64 {
        let mut log_soft: Vec<f64> = vec![0.0;self.num_outputs];
        util::log_soft_max(&self.preoutput, &mut log_soft, self.num_outputs);
        let loss: f64 = -1.0 * log_soft[label];
        return loss;
    }

    pub fn update_layer_params(&mut self, weight_grad: &Vec<f64>, bias_grad: &Vec<f64>, learning_rate: f64) {
        // Updating weight matrix
        for i in 0..self.num_inputs * self.num_outputs {
            self.weights[i] -= weight_grad[i] * learning_rate;
        }

        // Updating bias vector
        for i in 0..self.num_outputs {
            self.bias[i] -= bias_grad[i] * learning_rate;
        }
    }

    pub fn init_rand_params(&mut self) {
        init_rand_vector(&mut self.weights, self.num_inputs * self.num_outputs);

        for _i in 0..self.num_outputs {
            self.bias.push(0.0);
        }
    }

    pub fn clone(&self) -> Self {
        Layer {
            num_inputs: self.num_inputs,
            num_outputs: self.num_outputs,
            weights: self.weights.clone(),
            bias: self.bias.clone(),
            preoutput: self.preoutput.clone(),
            activation: self.activation,
            output: self.output.clone()
        }
    }

    pub fn get_num_inputs(&self) -> usize {
        self.num_inputs
    }

    pub fn get_num_outputs(&self) -> usize {
        self.num_outputs
    }

    pub fn get_output_vec(&self) -> Vec<f64> {
        self.output.clone()
    }

    pub fn get_weights(&self) -> Vec<f64> {
        self.weights.clone()
    }

}
