// main.rs

use serde::{Serialize, Deserialize};
use std::{collections::HashMap, sync::{Arc, Mutex}};
use warp::Filter;
use rand::Rng;
use tokio::task;
use rand::seq::SliceRandom;

// Model definition (simple linear model: y = Wx + b)
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Model {
    weights: Vec<f32>,
    bias: f32,
    status: String, // "initialized", "training", "ready"
}

impl Model {
    fn new(dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        Model {
            weights: (0..dim).map(|_| rng.gen_range(-0.5..0.5)).collect(),
            bias: rng.gen_range(-0.5..0.5),
            status: "initialized".to_string(),
        }
    }

    fn predict(&self, input: &[f32]) -> f32 {
        self.weights.iter().zip(input.iter())
            .map(|(w, x)| w * x)
            .sum::<f32>() + self.bias
    }

    fn train(&mut self, data: &[(Vec<f32>, f32)], epochs: usize, lr: f32) {
        self.status = "training".to_string();
        for _ in 0..epochs {
            for (x, y) in data {
                let pred = self.predict(x);
                let error = pred - y;
                for i in 0..self.weights.len() {
                    self.weights[i] -= lr * error * x[i];
                }
                self.bias -= lr * error;
            }
        }
        self.status = "ready".to_string();
    }

    fn average(models: &[Model]) -> Model {
        let n = models.len() as f32;
        let dim = models[0].weights.len();
        let mut avg_weights = vec![0.0; dim];
        let mut avg_bias = 0.0;
        for model in models {
            for (i, w) in model.weights.iter().enumerate() {
                avg_weights[i] += w;
            }
            avg_bias += model.bias;
        }
        for w in avg_weights.iter_mut() {
            *w /= n;
        }
        avg_bias /= n;
        Model { weights: avg_weights, bias: avg_bias, status: "ready".to_string() }
    }
}

// Server State
#[derive(Clone)]
struct ServerState {
    models: Arc<Mutex<HashMap<String, Model>>>,
    clients: Arc<Mutex<Vec<String>>>,
}

#[tokio::main]
async fn main() {
    let state = ServerState {
        models: Arc::new(Mutex::new(HashMap::new())),
        clients: Arc::new(Mutex::new(Vec::new())),
    };

    let state_filter = warp::any().map(move || state.clone());

    let register = warp::path!("register" / String)
        .and(warp::post())
        .and(state_filter.clone())
        .map(|model_name: String, state: ServerState| {
            let mut clients = state.clients.lock().unwrap();
            clients.push(model_name.clone());
            warp::reply::json(&format!("Registered for model {}", model_name))
        });

    let init = warp::path!("init" / String / usize)
        .and(warp::post())
        .and(state_filter.clone())
        .map(|model_name: String, dim: usize, state: ServerState| {
            let mut models = state.models.lock().unwrap();
            models.insert(model_name.clone(), Model::new(dim));
            warp::reply::json(&format!("Initialized model {}", model_name))
        });

    let get_model = warp::path!("get" / String)
        .and(warp::get())
        .and(state_filter.clone())
        .map(|model_name: String, state: ServerState| {
            let models = state.models.lock().unwrap();
            if let Some(model) = models.get(&model_name) {
                warp::reply::json(model)
            } else {
                warp::reply::json(&"Model not found")
            }
        });

    let train = warp::path!("train" / String / usize)
        .and(warp::post())
        .and(state_filter.clone())
        .and_then(start_training);

    let routes = register.or(init).or(get_model).or(train);

    println!("Starting server on 127.0.0.1:3030");
    task::spawn(warp::serve(routes).run(([127, 0, 0, 1], 3030)));

    // Simulate clients
    simulate_clients(5).await;
}

async fn start_training(model_name: String, rounds: usize, state: ServerState) -> Result<impl warp::Reply, warp::Rejection> {
    let mut global_model;
    {
        let models = state.models.lock().unwrap();
        global_model = models.get(&model_name).unwrap().clone();
    }

    for r in 0..rounds {
        println!("Training Round {}", r+1);

        // Simulate multiple clients training
        let mut local_models = vec![];
        for _ in 0..3 {
            let mut local_model = global_model.clone();
            let data = generate_dummy_data(28); // 28 inputs (like MNIST 28x28 flattened)
            local_model.train(&data, 1, 0.01);
            local_models.push(local_model);
        }

        // Federated Averaging
        let new_global = Model::average(&local_models);
        {
            let mut models = state.models.lock().unwrap();
            models.insert(model_name.clone(), new_global.clone());
        }
        global_model = new_global;
    }

    Ok(warp::reply::json(&"Training completed"))
}

// Simulated dummy data
fn generate_dummy_data(dim: usize) -> Vec<(Vec<f32>, f32)> {
    let mut rng = rand::thread_rng();
    (0..20).map(|_| {
        let input: Vec<f32> = (0..dim).map(|_| rng.gen_range(0.0..1.0)).collect();
        let label = if input.iter().sum::<f32>() > (dim as f32 / 2.0) { 1.0 } else { 0.0 };
        (input, label)
    }).collect()
}

// Simulate clients behavior
async fn simulate_clients(num_clients: usize) {
    for i in 0..num_clients {
        let client_id = format!("client{}", i);
        let register_url = format!("http://127.0.0.1:3030/register/{}", client_id);
        let init_url = format!("http://127.0.0.1:3030/init/{}/28", client_id);
        let client = reqwest::Client::new();
        
        // Register
        let _ = client.post(&register_url).send().await;
        // Initialize
        let _ = client.post(&init_url).send().await;

        println!("Client {} registered and initialized", client_id);
    }
}