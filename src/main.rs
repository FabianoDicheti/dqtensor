// mod lstm_test;
// mod mlp_test;
mod mlp_test2;

fn main() {


    // lstm_test::main();

    //  if let Err(e) = mlp_test::run_experiment() {
    //      eprintln!("Erro ao executar experimento Iris: {}", e);
    //  }
     if let Err(e) = mlp_test2::run_experiment_wine() {
         eprintln!("Erro ao executar experimento Wine: {}", e);
     }
 }
