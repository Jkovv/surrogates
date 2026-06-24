# Burn Wound Cytokine Surrogates (2D/3D)

Neural surrogate models for the cytokine dynamics of an agent-based burn-wound
simulation, split by spatial dimensionality (G=50):

- `surrogates_2D/` - planar (GxG) surrogates (U-Net, STA-LSTM, DeepONet, PINN, PI-DeepONet) and their pipeline.
- `surrogates_3D/` - volumetric (GxGxG) surrogates (U-Net, DeepONet) and their pipeline.

Each sub-project documents its own contents. 

Both share the same cytokine set:
`il8, il1, il6, il10, tnf, tgf`.


## Quick Start - Plug & Play

`demo.py` runs the whole pipeline end to end on a small corner of the data:

```
PIPELINE: carve a corner -> preprocess -> train a DeepONet (2d or 3d) -> metrics
```

A small VTK corner (16x16(x16), a few frames) is shipped in `demo_data/`, so you can clone the repo and run the demo with no data of your own:

```bash
pip install numpy scipy scikit-learn scikit-image pyvista
python demo.py --dims both --cytokine il8
```

This carves nothing new - it reads the bundled `demo_data/` corner, runs the real preprocessing, trains a light DeepONet, and prints metrics (Global R2,masked/unmasked RMSE, Dice, spatial correlation, SSIM). Results land in:
`demo_out/run_<stamp>__<cytokine>__c16/` as `results_2d.json`,
`results_3d.json`, and `summary.json`.


### Running on full data
Point the script at folders of `Step_*.vtk` frames:

```bash
python demo.py --dims both --cytokine il8 \
    --sim-2d /path/to/LatticeData/LatticeData(50x50) \
    --sim-3d /path/to/sim_output/LatticeData
```

### Regenerating the bundled demo corner

```bash
python demo.py --carve-demo \
    --sim-2d /path/to/LatticeData/LatticeData(50x50) \
    --sim-3d /path/to/sim_output/LatticeData
```

This keeps only the [:16,:16(,:16)] corner of a few frames, so no full-resolution simulation data is exposed.

### Useful flags

| Flag | Meaning |
|------|---------|
| `--dims {2d,3d,both}` | which pipeline(s) to run |
| `--cytokine il8` | cytokine to train on |
| `--corner 16` | corner edge length |
| `--epochs 60` | deepONet training no. epochs |
| `--max-frames 40` | cap frames per dimension |

## Data format
`Step_*.vtk` (CompuCell3D output)

## Full pipeline
The demo trains a light model on a corner. Full training (two cytokines, full grids, hyperparameter search) are included in `surrogates_2D/` and `surrogates_3D/`.
