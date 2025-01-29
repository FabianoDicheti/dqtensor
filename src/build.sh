set -e

echo "compiling Rust..."
cargo build --release

echo "compiling C++..."
cd src/cuda_project
nvcc -o cuda_exec main.cu

echo "Build Done!"