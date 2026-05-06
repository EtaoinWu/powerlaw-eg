# Power-Law stepsizes in min-max optimization

This repository contains the code for the NeurIPS 2026 submission #29265.

## Reproducibility

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
