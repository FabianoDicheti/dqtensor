mod lstm_test;
mod mlp_test;
mod mlp_test2;
mod mamba_test;

fn main() {

     if let Err(e) = mamba_test::main() {
         eprintln!("mamba test: {}", e);
     }

    //  if let Err(e) = mlp_test::run_experiment() {
    //      eprintln!("Experiment MLP Iris: {}", e);
    //  }

    //  if let Err(e) = mlp_test2::run_experiment_wine() {
    //      eprintln!("Experiment MLP Wine: {}", e);
    //  }

    //   if let Err(e) = lstm_test::main() {
    //      eprintln!("Experiment LSTM Jena Climate: {}", e);
    //  }
 }
