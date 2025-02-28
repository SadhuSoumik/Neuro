use rand::Rng;

#[derive(Debug, Clone)]
struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    fn new(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    fn dot(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows, "Matrix dimensions do not match for multiplication");
        let mut result = Matrix::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i * self.cols + k] * other.data[k * other.cols + j];
                }
                result.data[i * result.cols + j] = sum;
            }
        }
        result
    }

    fn add(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows, "Matrix dimensions do not match for addition");
        assert_eq!(self.cols, other.cols, "Matrix dimensions do not match for addition");
        let data = self.data.iter().zip(other.data.iter()).map(|(a, b)| a + b).collect();
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    fn subtract(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows, "Matrix dimensions do not match for subtraction");
        assert_eq!(self.cols, other.cols, "Matrix dimensions do not match for subtraction");
        let data = self.data.iter().zip(other.data.iter()).map(|(a, b)| a - b).collect();
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    fn transpose(&self) -> Matrix {
        let mut result = Matrix::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[j * result.cols + i] = self.data[i * self.cols + j];
            }
        }
        result
    }

    // Apply a function to each element
    fn apply<F>(&self, f: F) -> Matrix 
    where F: Fn(f64) -> f64 
    {
        let data = self.data.iter().map(|x| f(*x)).collect();
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    // Element-wise multiplication
    fn hadamard(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows, "Matrix dimensions do not match for Hadamard product");
        assert_eq!(self.cols, other.cols, "Matrix dimensions do not match for Hadamard product");
        let data = self.data.iter().zip(other.data.iter()).map(|(a, b)| a * b).collect();
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
    
    // Debug print the matrix dimensions
    fn debug_dims(&self, name: &str) {
        println!("Matrix {}: {}x{}", name, self.rows, self.cols);
    }
}

#[derive(Debug, Clone)]
struct Layer {
    weights: Matrix,
    biases: Matrix,
    // Store the transpose of weights for efficiency
    weights_t: Matrix,
}

impl Layer {
    fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        // Xavier/Glorot initialization
        let scale = (2.0 / (input_size + output_size) as f64).sqrt();
        
        let weights = Matrix {
            rows: input_size,
            cols: output_size,
            data: (0..input_size * output_size)
                .map(|_| rng.gen_range(-scale..scale))
                .collect(),
        };
        
        let weights_t = weights.transpose();
        
        let biases = Matrix {
            rows: 1,
            cols: output_size,
            data: vec![0.0; output_size],
        };
        
        Layer { weights, biases, weights_t }
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_derivative(s: f64) -> f64 {
    // s is already the sigmoid output
    s * (1.0 - s)
}

struct SGD {
    learning_rate: f64,
}

impl SGD {
    fn new(learning_rate: f64) -> Self {
        SGD { learning_rate }
    }

    fn update(&self, layer: &mut Layer, weight_gradients: &Matrix, bias_gradients: &Matrix) {
        // Ensure dimensions match before updating
        assert_eq!(layer.weights.rows, weight_gradients.rows, 
            "Weight gradient rows mismatch, expected {}, got {}", 
            layer.weights.rows, weight_gradients.rows);
        assert_eq!(layer.weights.cols, weight_gradients.cols, 
            "Weight gradient cols mismatch, expected {}, got {}", 
            layer.weights.cols, weight_gradients.cols);
        
        for i in 0..layer.weights.data.len() {
            layer.weights.data[i] -= self.learning_rate * weight_gradients.data[i];
        }
        
        // Update the transpose as well
        layer.weights_t = layer.weights.transpose();
        
        for i in 0..layer.biases.data.len() {
            layer.biases.data[i] -= self.learning_rate * bias_gradients.data[i];
        }
    }
}

#[derive(Debug)]
struct NeuralNetwork {
    layers: Vec<Layer>,
}

impl NeuralNetwork {
    fn new(sizes: Vec<usize>) -> Self {
        let mut layers = Vec::new();
        for i in 0..sizes.len() - 1 {
            layers.push(Layer::new(sizes[i], sizes[i + 1]));
        }
        NeuralNetwork { layers }
    }

    fn forward(&self, input: &Matrix) -> Vec<Matrix> {
        let mut activations = vec![input.clone()];
        let mut current = input.clone();
        
        for layer in &self.layers {
            // Calculate z = w^T·a + b (weight * activation + bias)
            let weighted_input = layer.weights_t.dot(&current);
            
            // Add biases
            let mut z = weighted_input.clone();
            for i in 0..z.rows {
                for j in 0..z.cols {
                    z.data[i * z.cols + j] += layer.biases.data[j];
                }
            }
            
            // Apply activation function: a = sigmoid(z)
            current = z.apply(sigmoid);
            activations.push(current.clone());
        }
        
        activations
    }

    fn train(&mut self, inputs: &Vec<Matrix>, targets: &Vec<Matrix>, epochs: usize, optimizer: &SGD) {
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            
            for (input, target) in inputs.iter().zip(targets.iter()) {
                // Forward pass
                let activations = self.forward(input);
                let output = activations.last().unwrap();
                
                // Calculate loss and output error
                let error = output.subtract(target);
                total_loss += error.data.iter().map(|e| e * e).sum::<f64>() / error.data.len() as f64;
                
                // Backpropagation
                let mut deltas = Vec::with_capacity(self.layers.len());
                
                // Calculate output layer error (delta = (y_hat - y) * sigmoid'(a))
                let output_layer_delta = error.hadamard(&output.apply(sigmoid_derivative));
                deltas.push(output_layer_delta);
                
                // Propagate error backward through the network
                for l in (0..self.layers.len() - 1).rev() {
                    let next_delta = &deltas[deltas.len() - 1];
                    let current_activation = &activations[l + 1];
                    let next_weights = &self.layers[l + 1].weights;
                    
                    // delta = (w_{l+1} · delta_{l+1}) ⊙ sigmoid'(a_l)
                    let delta = next_weights.dot(next_delta).hadamard(
                        &current_activation.apply(sigmoid_derivative)
                    );
                    
                    deltas.push(delta);
                }
                
                // Reverse deltas to match layer indices (first is output layer, last is first hidden)
                deltas.reverse();
                
                // Update weights and biases for each layer
                for l in 0..self.layers.len() {
                    let delta = &deltas[l];
                    let prev_activation = &activations[l];
                    
                    // Calculate gradients
                    // dW = a_{l-1} · delta_l^T
                    let weight_gradients = Matrix {
                        rows: self.layers[l].weights.rows,
                        cols: self.layers[l].weights.cols,
                        data: Vec::with_capacity(self.layers[l].weights.data.len()),
                    };
                    
                    // Manually calculate weight gradients to ensure correct dimensions
                    let mut gradient_data = Vec::with_capacity(self.layers[l].weights.data.len());
                    for i in 0..prev_activation.rows {
                        for j in 0..delta.rows {
                            let mut sum = 0.0;
                            for k in 0..prev_activation.cols {
                                // If single column matrices, the calculation simplifies
                                if prev_activation.cols == 1 && delta.cols == 1 {
                                    sum += prev_activation.data[i] * delta.data[j];
                                } else {
                                    sum += prev_activation.data[i * prev_activation.cols + k] * 
                                           delta.data[j * delta.cols + k];
                                }
                            }
                            gradient_data.push(sum);
                        }
                    }
                    
                    let weight_gradients = Matrix {
                        rows: self.layers[l].weights.rows,
                        cols: self.layers[l].weights.cols,
                        data: gradient_data,
                    };
                    
                    // For biases, sum delta across batch dimension
                    let mut bias_gradient_data = vec![0.0; delta.rows];
                    for i in 0..delta.rows {
                        for j in 0..delta.cols {
                            bias_gradient_data[i] += delta.data[i * delta.cols + j];
                        }
                    }
                    
                    let bias_gradients = Matrix {
                        rows: 1,
                        cols: delta.rows,
                        data: bias_gradient_data,
                    };
                    
                    // Update weights and biases
                    optimizer.update(&mut self.layers[l], &weight_gradients, &bias_gradients);
                }
            }
            
            if epoch % 100 == 0 {
                println!("Epoch {}: Loss = {}", epoch, total_loss / inputs.len() as f64);
            }
        }
    }
}

fn main() {
    // XOR dataset
    let inputs = vec![
        Matrix { rows: 2, cols: 1, data: vec![0.0, 0.0] },
        Matrix { rows: 2, cols: 1, data: vec![0.0, 1.0] },
        Matrix { rows: 2, cols: 1, data: vec![1.0, 0.0] },
        Matrix { rows: 2, cols: 1, data: vec![1.0, 1.0] },
    ];
    let targets = vec![
        Matrix { rows: 1, cols: 1, data: vec![0.0] },
        Matrix { rows: 1, cols: 1, data: vec![1.0] },
        Matrix { rows: 1, cols: 1, data: vec![1.0] },
        Matrix { rows: 1, cols: 1, data: vec![0.0] },
    ];

    // Initialize and train the network
    let mut nn = NeuralNetwork::new(vec![2, 4, 1]);
    let optimizer = SGD::new(0.1);
    nn.train(&inputs, &targets, 2000, &optimizer);

    // Test the trained network
    println!("\nTesting the trained network:");
    for input in &inputs {
        let activations = nn.forward(input);
        let output = activations.last().unwrap();
        println!("Input: {:?}, Output: {:?}", input.data, output.data);
    }
}
