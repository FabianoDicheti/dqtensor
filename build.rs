use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let kernels = vec![
        "conv1d.cu",
        "discretize_zoh.cu",
        "expm_mul.cu",
        "generate_dynamic_params.cu",
        "apply_gate.cu",
        "ssm_kernel_conv.cu",
        "simulate_scan.cu",
    ];

    for kernel in kernels {
        let src_path = PathBuf::from("cuda").join(kernel);
        let ptx_name = kernel.replace(".cu", ".ptx");
        let dst_path = out_dir.join(ptx_name);

        let status = Command::new("nvcc")
            .args(["-ptx", src_path.to_str().unwrap(), "-o", dst_path.to_str().unwrap()])
            .status()
            .expect("Failed to run nvcc");

        assert!(status.success(), "nvcc failed for {}", kernel);

        // Informa ao Cargo que deve recompilar se esse arquivo mudar
        println!("cargo:rerun-if-changed=cuda/{}", kernel);
    }
} 