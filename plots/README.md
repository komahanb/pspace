# Plots

Reusable visualisation scripts showcasing basis functions, quadrature nodes, and
polynomial behaviour for `pspace` coordinate systems.

## `plot_basis_gallery.py`

Generates a gallery containing:

- 1D curves for each axis' orthonormal basis (ψ-functions).
- 2D heatmaps for pairwise ψ_i ⊗ ψ_j combinations.
- Quadrature node visualisations.
- Sample PolyFunction slices/contours.

```bash
python3 plots/plot_basis_gallery.py \
    --basis-type tensor \
    --num-coords 3 \
    --max-degree 3 \
    --output-dir plots/output/gallery
```

The script builds both numeric and plotting coordinate-system duals, ensuring the
plots align with the core computations. Additional plotting scripts can follow the
same pattern, reusing helpers from `plots/_helpers.py`.
