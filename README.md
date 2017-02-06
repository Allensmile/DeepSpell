# DeepSpell
Deep Learning based Speller

## Quick Start
Clone DeepSpell Git repository:

```git clone https://github.com/surmenok/DeepSpell.git```

### CPU
* Install [Docker](https://www.docker.com/)
* Run `./build.sh`
* Run `docker run --name=deepspell-cpu -it deepspell-cpu`

### GPU
Requires CUDA-compatible graphics card.

* Install [NVIDIA docker](https://www.docker.com/)
* Run `./build.sh gpu`
* Run `nvidia-docker run --name=deepspell-gpu -it deepspell-gpu`

## Documentation
[Deep Spelling](https://medium.com/@majortal/deep-spelling-9ffef96a24f6#.2c9pu8nlm) by Tal Weiss
