# Water Models

For water detectors, Olympus doesn't track each photon. It uses two trained networks instead. This page describes the models that ship with Prometheus, how to choose one, and why the sea water model changed.

Each model has two parts:

- **Counts model**: predicts the fraction of a source's photons that reach a module, given the distance and the viewing angle.
- **Shape model**: predicts when those photons arrive.

## Available Models

Prometheus selects a model from the `Medium` entry in your geofile. All model files are in `resources/olympus_resources`.

| Model files | Used for | Emission | Status |
| --- | --- | --- | --- |
| `photon_arrival_time_*` | `Medium: water` | Not covered on this page | Unchanged |
| `km3net_sea_cherenkov_*` | `Medium: mediterranean` | Cherenkov cone of a cascade | Default |
| `km3net_sea_isotropic_*` | Select it yourself | Isotropic | Same water, smoother light pattern |
| `antares_*_low_E` | Select it yourself | Not applicable | Kept to reproduce older results. Don't use it for new work. |

!!! warning
    If your geofile uses `Medium: mediterranean`, your results change compared with the `antares_*_low_E` model. Distant modules see far less light, and the light pattern around each source is different.

## Choosing a Model

To override the default, set both file names in your configuration before you create the `Prometheus` object:

```python
from prometheus import config

config.photon_propagator.olympus.paths.flow = "km3net_sea_isotropic_nflow_params.pickle"
config.photon_propagator.olympus.paths.counts = "km3net_sea_isotropic_counts_params.pickle"
```

Use the Cherenkov model unless you have a reason not to. The isotropic model is useful when you compare with results that were produced with it, or when you want to separate the effect of the light pattern from the effect of the total light.

## Discussion

### What Was Wrong with the Old Model

The propagator multiplies the output of the counts model by every photon that a source emits. It looks up that output using the viewing angle, which is the angle between the source axis and the line from the source to the module.

The old generator instead grouped photons by the angle at which each photon left the source. These two angles describe different things, so the `antares_*_low_E` model answers a different question from the one that the propagator asks. As a result, its light peaks at 0 degrees instead of near the Cherenkov angle of 42 degrees.

The old model also predicts too much light at large distances. The following table compares its average over all viewing angles with the new Cherenkov model:

| Distance | Light in the old model relative to the new model |
| --- | --- |
| 10 m | 1.1 times |
| 30 m | 3.4 times |
| 50 m | 6.8 times |
| 100 m | 32 times |

At its worst, the old model puts about 640 times more light on a module than the module's size allows, even if a source sent all of its photons into one steradian.

### Cherenkov and Isotropic Emission

Both new models describe the same water and use the viewing angle. They differ in how a source emits light:

- The **Cherenkov** model concentrates light near 42 degrees around the source axis, as a cascade does.
- The **isotropic** model spreads the same light evenly, so it has the correct total but not the correct pattern.

The difference depends on where an event points. The following results are from a small test of 10 electron-neutrino events between 10 GeV and 100 GeV in the ORCA geometry:

- The ratio of Cherenkov hits to isotropic hits ranges from 0.41 to 3.11 per event. These ratios follow the values predicted from the two networks and the event directions.
- The total ratio is 1.55 for these events, because most of them point into the detector.
- Averaged over random directions at the same positions, the predicted ratio is 1.06. A value close to 1 is what light conservation requires.

If you only need the total light from many events with random directions, the two models are close. If you depend on which modules see light, or on directions, use the Cherenkov model. This test checks that the model runs and behaves as expected. It doesn't validate the physics.

### Why the Training Labels Are Pooled

Far from a source, the light in a training row can depend on a few scattered photons with large weights. Such a photon raises the row's label and also lowers its effective count, which is the number of photons that the label effectively rests on.

A fit weighted by the effective count therefore follows the rows that missed these photons, and the network loses light. Without any correction, the Cherenkov counts model has only 43 percent of the expected light at 73 m.

To avoid this, `02_train_antares_nflow.py` averages neighboring rows with `pool_counts_rows` before the fit. The size of each averaging window depends only on the distance. A window that grew until its effective count looked large enough would stop early on exactly the rows that missed the heavy photons, and the problem would return.

### Known Limitations

Keep these limitations in mind when you interpret results:

- **Cherenkov model beyond about 150 m**: the training data is too noisy to test the model well. Between 148 m and 300 m, the network has 0.65 ± 0.29 of the light in the labels.
- **Cone peak**: pooling and the small network blur the peak. At the 42.2 degree row, the model is 17 percent low at 21 m, 30 percent low at 30 m, and about 50 percent low at 51 m to 73 m. The total light is preserved, so this mostly affects modules within about one degree of the peak.
- **Arrival times**: the shape model trains and loads, but its timing isn't yet compared with the training data.
- **Isotropic model above about 100 m**: its labels above 118 m are extrapolated and leave out scattered light. The Cherenkov labels are 3 to 10 times higher between 104 m and 300 m. See `far_rows_extrapolated` in `km3net_sea_isotropic_training_meta.json`.
- **Isotropic training path**: the isotropic files come from merged training runs without pooling, so the current scripts don't reproduce them exactly.

## Training Your Own Model

The scripts are in the [examples/water_model directory](https://github.com/Harvard-Neutrino/prometheus/tree/main/examples/water_model). Run them from the root of the repository, in an environment that imports Hyperion from the same checkout.

First, generate the training data. The following settings produce the data for the shipped Cherenkov model, which takes about 13 hours on four CPU cores:

```sh
python examples/water_model/01_generate_photon_mc.py --emission cherenkov --refine-deg 8 --n-photons 500000 --n-distances 40 --n-angle-bins 18 --d-min 0.31 --d-max 300 --aim-fraction 0.25 --chunk-size 500000 --max-time 5000 --seed 20261104 --out-dir examples/output/water_model
```

To generate data for an isotropic model, use `--emission isotropic` instead.

Then train the models. The output directory keeps your models apart from the shipped files, which have the same names:

```sh
python examples/water_model/02_train_antares_nflow.py --data-dir examples/output/water_model --out-dir examples/output/water_model/models
```

This script checks the counts model before it saves it. If the model fails, the script saves it with a `.FAILED` suffix and exits with status 1.

To use your models in a simulation, set `config.photon_propagator.olympus.paths.location` to the output directory, and set the file names as described in [Choosing a Model](#choosing-a-model).

To check any counts model yourself, run:

```sh
python examples/water_model/03_check_photon_model.py --counts examples/output/water_model/models/km3net_sea_cherenkov_counts_params.pickle --counts-data examples/output/water_model/counts_data.npz
```

The script runs the following checks. Checks 5 and 6 need `--counts-data`.

| Check | What it tests |
| --- | --- |
| 1. Geometric bound | No module receives more light than its size allows. |
| 2. Cone convention | The light is flat, or it peaks between 20 and 65 degrees. |
| 3. Attenuation | Light falls steadily beyond 40 m, at a rate that matches the water's absorption. |
| 4. Training domain | The model is trained over all the distances that you need. |
| 5. Training rows | The model fits the labels that it was trained on. |
| 6. Light conservation | The network matches the raw labels, averaged over viewing angle, in bands of distance. |
