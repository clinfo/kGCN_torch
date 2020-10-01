curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

rustup toolchain install nightly
rustup default nightly
pip install torchex setuptools_rust
pip install scipy numpy networkx

pip install git+https://github.com/0h-n0/thdbonas.git
git+https://github.com/0h-n0/frontier_graph.git
pip install git+https://github.com/0h-n0/inferno.git

CUDA="cu102"
pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-geometric

