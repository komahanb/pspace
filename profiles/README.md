# Profiles

Benchmark utilities for the `pspace` decomposition operators. These
scripts do **not** check numerical correctness; they are meant to help
characterise runtime performance as the number of variables and the
basis size grow.

## `decomposition_profiles.py`

Runs rank-1 (vector) and rank-2 (matrix) decompositions across each
inner-product backend (`numerical`, `symbolic`, `analytic`). Results are
saved as a CSV table along with optional bar plots.

```bash
python3 profiles/decomposition_profiles.py \
    --trials 3 \
    --num-coords 4 \
    --max-degree 2 \
    --output-dir profiles/output
```

Key options:

| Flag | Meaning |
| --- | --- |
| `--trials` | Number of repetitions per combination (default `3`). |
| `--num-coords` | How many coordinates/variables to include. |
| `--max-degree` | Maximum monomial degree per coordinate. |
| `--dense` | Use dense assembly instead of sparsity-aware masks. |
| `--no-plots` | Skip Matplotlib plots (CSV only). |
| `--seed` | Reproducibility seed (default `2025`). |

Outputs land in the requested directory as:

* `decomposition_timings.csv` – raw measurement table.
* `[vector|matrix]_timings.png` – optional bar plots (if plotting is enabled).
