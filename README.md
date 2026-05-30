# Power-Law stepsizes in min-max optimization

This repository contains the code for the the manuscript, titled *Accelerating Min-Max Optimization via Power-Law Stepsizes*, by [Yue Wu](https://wuy.me), [Weiqiang Zheng](https://weiqiang-zheng.com/), [Yang Cai](https://www.cs.yale.edu/homes/cai/), and [Haipeng Luo](https://haipeng-luo.net/).

## Setup

The code manages its environment using [Pixi](https://pixi.prefix.dev/). To set up the environment, run:

```bash
pixi install
pixi install --environment dev
```

## Linting

The code is linted using [ruff](https://github.com/astral-sh/ruff), [pyright](https://github.com/microsoft/pyright), and [mypy](https://www.mypy-lang.org/). 
The unit tests are performed with [pytest](https://docs.pytest.org/).
These packages are installed with the development environment. To run the linters, use:

```bash
pixi run --environment dev ruff check src
pixi run --environment dev ruff format --check src
pixi run --environment dev pyright src
pixi run --environment dev mypy src
pixi run --environment dev pytest .
```

## Experiments

The code for the experiments used in the paper is located in the `experiments/` directory, as ipython notebook files. You can run all cells in the notebooks in order ("run all") to reproduce the results in the paper.

The random seeds for the experiments are set fixed to ensure reproducibility. The output of the experiments in the `results/` directory should be identical to the plots in the paper.
