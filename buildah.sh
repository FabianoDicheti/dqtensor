set -e

container=$(buildah from nvidia/cuda:12.2.2-devel-ubuntu20.04)

buildah run $container apt update -y
buildah run $container apt install -y curl git build-essential cmake python3

buildah run $container curl --proto '=https' --tlsv1.2 -sSf http://sh.rustup.rs | sh -s -- -y
buildah run $container bash -c "source $HOME/.cargo/env"

buildah copy $container . /app
buildah config --workingdir /app $container

buildah run $container chmod +x /app/src/build.sh

buildah run $container /app/src/build.sh

buildah config --cmd "/app/target/release/binario_rust" $container

buildah commit $container m_project:latest

echo "Container Image Builded"
