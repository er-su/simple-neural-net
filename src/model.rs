use layer::Layer;
use rand::thread_rng;
use rand::seq::SliceRandom;

pub(crate) mod layer;

pub struct Hyperparams {
    pub epochs: usize,
    pub batch_size: usize,
    pub learn_rate: f64,
}

pub struct Model {
    num_inputs: usize,
    num_outputs: usize,
    layers: Vec<layer::Layer>,
    weights_grad: Vec<Vec<f64>>,
    biases_grad: Vec<Vec<f64>>,
}

impl Model {
    pub fn new(hidden_act_func: fn(&Vec<f64>, &mut Vec<f64>, usize), output_act_fun: fn(&Vec<f64>, &mut Vec<f64>, usize)) -> Self {
        let mut new_model = Model {
            num_inputs: 784,
            num_outputs: 10,
            layers: Vec::new(),
            weights_grad: Vec::new(),
            biases_grad: Vec::new(),
        };

        new_model.layers.push(Layer::new(new_model.num_inputs, 512, hidden_act_func));
        new_model.layers.push(Layer::new(512, 512, hidden_act_func));
        new_model.layers.push(Layer::new(512, new_model.num_outputs, output_act_fun));

        let layer_len = new_model.layers.len();

        //new_model.weights_grad.reserve(layer_len);
        //new_model.biases_grad.reserve(layer_len);
        // FIX ALLOCATION LAYER


        for i in 0..layer_len {
            new_model.weights_grad.push(vec![0.0; new_model.layers[i].get_num_inputs() * new_model.layers[i].get_num_outputs()]);
            new_model.biases_grad.push(vec![0.0; new_model.layers[i].get_num_outputs()]);
        }
        
        new_model.zero_grad();

        return new_model;        
    }

    pub fn forward(&mut self, input: &Vec<f64>, size: usize) -> Vec<f64> {
        assert_eq!(size, self.num_inputs);

        let mut prev_input: Vec<f64> = Vec::new();
        prev_input.clone_from(&input);

        let mut input_size = size;

        for i in 0..self.layers.len() {
            prev_input = self.layers[i].forward(prev_input, input_size);
            input_size = self.layers[i].get_num_outputs();
        }

        return prev_input;
    }

    pub fn train(&mut self, train_images: &Vec<Vec<f64>>, image_length: usize, count: usize, labels: &Vec<usize>, p: &Hyperparams) {
        let mut indicies: Vec<usize> = (0..count).collect();

        let num_batches: usize = (count - 1)/p.batch_size + 1;
    
        for _e in 0..p.epochs {
            indicies.shuffle(&mut thread_rng());
            let mut global_count: usize = 0;
            for b in 0..num_batches {
                let mut print_loss: bool = false;
                let mut batch_loss:f64 = 0.0;

                if b % 100 == 0 {
                    print_loss = true;
                    print!("/{num_batches} loss: ");
                }

                let mut i = 0;
                while i < p.batch_size && global_count < count {
                    let image_index = indicies[global_count];
                    let loss = self.backpropagate(&train_images[image_index], image_length, labels[image_index]);

                    batch_loss += loss;

                    i+=1;
                    global_count+=1;
                }

                if print_loss {
                    println!("{batch_loss}");
                }

                self.update_model_params(p.learn_rate);
                self.zero_grad();
            } 
        }
    }

    fn backpropagate(&mut self, image: &Vec<f64>, image_length: usize, label: usize) -> f64 {
        self.forward(&image, image_length);

        let loss = self.layers[self.layers.len() - 1].calc_loss(label);

        let last_depth = self.layers.len() - 1;

        let mut prev_error_layer = self.layers[last_depth].calc_last_error(label);
        self.single_bias_grad_update(last_depth, &prev_error_layer, self.layers[last_depth].get_num_outputs());
        self.single_weights_grad_update(last_depth, &prev_error_layer, &self.layers[last_depth - 1].get_output_vec(), self.layers[last_depth].get_num_outputs(), self.layers[last_depth].get_num_inputs());

        let mut layer_error:Vec<f64>;

        let mut i: usize = self.layers.len() - 2;
        while i >= 1 {
            layer_error = self.layers[i].calc_layer_error(&prev_error_layer, &self.layers[i + 1].get_weights(), self.layers[i+1].get_num_inputs(), self.layers[i+1].get_num_outputs());
            self.single_bias_grad_update(i, &layer_error, self.layers[i].get_num_outputs());
            self.single_weights_grad_update(i, &layer_error, &self.layers[i-1].get_output_vec(), self.layers[i].get_num_outputs(), self.layers[i].get_num_inputs());
            prev_error_layer.clear();
            prev_error_layer = layer_error.clone();
            i -= 1;
        }

        layer_error = self.layers[0].calc_layer_error(&prev_error_layer, &self.layers[1].get_weights(), self.layers[1].get_num_inputs(), self.layers[1].get_num_outputs());
        self.single_bias_grad_update(0, &layer_error, self.layers[0].get_num_outputs());
        self.single_weights_grad_update(0, &layer_error, &image, self.layers[0].get_num_outputs(), self.layers[0].get_num_inputs());
    
        return loss;
    }

    fn single_bias_grad_update(&mut self, depth: usize, layer_error: &Vec<f64>, bias_length: usize) {
        for i in 0..bias_length {
            self.biases_grad[depth][i] += layer_error[i];
        }
    }

    fn single_weights_grad_update(&mut self, depth: usize, layer_error: &Vec<f64>, layer_input: &Vec<f64>, layer_length: usize, input_length: usize) {
        assert_eq!(self.layers[depth].get_num_outputs(), layer_length);
        assert_eq!(self.layers[depth].get_num_inputs(), input_length);

        let mut single_weight_grad: Vec<f64> = vec![0.0; layer_length * input_length];

        for i in 0..layer_length {
            for j in 0..input_length {
                single_weight_grad[i * input_length + j] = (layer_error[i] * layer_input[j]);
            }
        }

        for i in 0..layer_length * input_length {
            self.weights_grad[depth][i] += single_weight_grad[i];
        }
    }

    fn zero_grad(&mut self) {
        for i in 0..self.layers.len() {
            for j in 0..(self.layers[i].get_num_inputs() * self.layers[i].get_num_outputs()) {
                self.weights_grad[i][j] = 0.0;
            }

            for j in 0..self.layers[i].get_num_outputs() {
                self.biases_grad[i][j] = 0.0;
            }
        }
    } 

    fn update_model_params(&mut self,learning_rate: f64) {
        for i in 0..self.layers.len() {
            self.layers[i].update_layer_params(&self.weights_grad[i], &self.biases_grad[i], learning_rate);
        }
    }

    pub fn test(&mut self, test_images: &Vec<Vec<f64>>, image_length: usize, count: usize, labels: &Vec<usize>) {
        let mut correct = 0;
        for i in 0..count {
            let outputs = self.forward(&test_images[i], image_length);

            let mut max_index = 0;
            let mut max_val = outputs[0];

            for j in 1..self.num_outputs {
                if outputs[j] > max_val  {
                    max_index = j;
                    max_val = outputs[j];
                }
            }
            if max_index == labels[i] {
                correct += 1;
            }

        }
        println!("{correct}/{count} correct");
    }
}