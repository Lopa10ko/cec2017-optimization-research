# CEC-2017 Optimization Research

Comparative study of numerical optimization methods on the [CEC-2017](https://github.com/P-N-Suganthan/CEC2017-BoundContrained) single-objective bound-constrained benchmark suite. The benchmark is provided by [cec2017-py](https://github.com/tilleyd/cec2017-py); this repository implements reproducible experiments (visualization, classical and metaheuristic optimizers, statistical comparison) with a clean Python package layout and `uv` for dependency management.

---

## Stage 1: Benchmark visualization

**Interactive f1-f10 explorer (GitHub Pages):** [https://lopa10ko.github.io/cec2017-optimization-research/cec_f1_f10.html](https://lopa10ko.github.io/cec2017-optimization-research/cec_f1_f10.html)


**Static gallery (f1–f10):**

| | |
|:---:|:---:|
| <img src="outputs/stage1/f1.png" width="450" alt="f1"/> | <img src="outputs/stage1/f2.png" width="450" alt="f2"/> |
| <img src="outputs/stage1/f3.png" width="450" alt="f3"/> | <img src="outputs/stage1/f4.png" width="450" alt="f4"/> |
| <img src="outputs/stage1/f5.png" width="450" alt="f5"/> | <img src="outputs/stage1/f6.png" width="450" alt="f6"/> |
| <img src="outputs/stage1/f7.png" width="450" alt="f7"/> | <img src="outputs/stage1/f8.png" width="450" alt="f8"/> |
| <img src="outputs/stage1/f9.png" width="450" alt="f9"/> | <img src="outputs/stage1/f10.png" width="450" alt="f10"/> |


### Project structure

Layout adapted from the [cookiecutter-research-project](https://github.com/aeturrell/cookiecutter-research-project) data-science template. Included paths match this study; empty template folders (`data/`, `logs/`, `models/`, `paper/`) are omitted because artifacts live under `outputs/` and the benchmark is installed as a dependency.

```text
cec2017-optimization-research/
├── pyproject.toml
├── README.md
├── notebooks/
│   └── experiments.ipynb
├── outputs/
│   ├── stage1/                 # 3D surface plots (Stage 1)
│   ├── stage2/                 # SciPy L-BFGS-B tables (planned)
│   ├── stage3/                 # PSO / GA tables + boxplots (planned)
│   └── stage4/                 # Nemenyi test outputs (planned)
└── src/
    └── cec2017/
        ├── benchmark/
        └── visualization/
```

| Template path | This repo |
|---------------|-----------|
| `pyproject.toml`, `uv` | Yes |
| `notebooks/`, `outputs/`, `src/<package>/` | Yes (`src/cec2017/`) |
| `data/raw`, `intermediate`, `processed` | No — CEC-2017 from git dependency |
| `logs/`, `models/`, `paper/`, `slides/` | No — not required for this coursework |

### Reproducibility (Stage 1)

Requires [uv](https://docs.astral.sh/uv/).

```bash
cd cec2017-optimization-research
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv sync

# then go to notebooks/experiments.ipynb
# or run via:
uv run jupyter notebook notebooks/experiments.ipynb
```

---

## Stage 2: SciPy L-BFGS-B (planned)

10 runs per function f1–f28 at $D = 10$ with random starts; table of best $f_{\min}$ in README and CSV under `outputs/stage2/`.

## Stage 3: PSO and GA (planned)

PySwarms GlobalBestPSO and DEAP/PyGAD genetic algorithms; timing and boxplots under `outputs/stage3/`.

## Stage 4: Statistical comparison (planned)

Nemenyi post-hoc test across methods from Stages 2–3; figures under `outputs/stage4/`.

---

## License

See [LICENSE](LICENSE).
