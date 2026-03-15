# PPC Table Files Reference

This document describes each file in a PPC table directory, where it is read in
the source code, and what is known (and unknown) about the equivalent quantities
for realistic deep-water neutrino telescope simulations.

All code references are to `resources/PPC_executables/PPC/ini.cxx`.

---

## `cfg.txt`

**Read at:** line 262–320

**Format:** four values on successive lines

| Line | Parameter | Description |
|------|-----------|-------------|
| 1 | `oversize` (`xR`) | DOM radius oversize factor relative to nominal 16.32 cm. Scales the geometric cross-section to account for photon collection efficiency. |
| 2 | `efficiency` (`eff`) | Overall DOM efficiency multiplier applied on top of the QE curve in `wv.dat`. |
| 3 | `sam_fraction` (`sf`) | Fraction of scattering described by the simplified angular model (SAM) vs Henyey-Greenstein. See Section 5 of the [SPICE paper](https://user-web.icecube.wisc.edu/~dima/work/WISC/ppc/spice/paper/a.pdf). |
| 4 | `g` | Mean cosine of the photon scattering angle ⟨cos θ⟩. Used in `sca = b_e * l_a / (1 - g)` to convert the effective scattering coefficient to the geometric mean free path. |

**PPC formula context:**
```cpp
// ini.cxx:591
float sca = be[j] * l_a / (1 - d.g);
```

---

## `icemodel.dat`

**Read at:** lines 535–549; used in absorption/scattering loop at lines 588–592

**Format:** one depth layer per line, four columns

| Column | Quantity | Units |
|--------|----------|-------|
| 0 | Layer center depth (positive = deeper) | m |
| 1 | Effective scattering coefficient b_e(400 nm) | m⁻¹ |
| 2 | Dust absorption coefficient a_dust(400 nm) | m⁻¹ |
| 3 | Δτ = T(d) − T(1730 m), temperature deviation | dimensionless |

Layers must be uniformly spaced (tolerance 1×10⁻⁵, line 555).

**PPC formula context:**
```cpp
// ini.cxx:591
// be[j] = column 1,  ba[j] = column 2,  td[j] = column 3
float sca = be[j] * l_a / (1 - d.g);
float abs = (D * ba[j] + E) * l_k + ABl * (1 + 0.01 * td[j]);
```

Note: `ba` (column 2) is **only the dust component** of absorption. Total absorption
also includes pure-water/ice contributions from `icemodel.par` (A, B, E terms).

---

## `icemodel.par`

**Read at:** lines 520–531; parameters used in wavelength scaling at lines 583–591

**Format:** six rows, each with `value  uncertainty`. Uncertainties are stored but
not used by PPC.

| Row | SPICE name | AMANDA name | Description |
|-----|-----------|-------------|-------------|
| 1 | α | α | Scattering power-law slope: `b(λ) = b(400) * (λ/400)^(-α)` |
| 2 | κ | κ | Absorption power-law slope for dust: `a_dust(λ) ∝ (λ/400)^(-κ)` |
| 3 | A | A_IR | Pure-water/ice absorption amplitude in `ABl = A * exp(-B/λ)` |
| 4 | B | λ₀ | Pure-water/ice absorption scale wavelength [nm] |
| 5 | D | — | Temperature correction coefficient for scattering |
| 6 | E | — | Temperature correction offset for absorption |

**PPC formula context:**
```cpp
// ini.cxx:583-591  (wv0 = 400 nm reference wavelength)
float l_a = pow(wva / wv0, -a);      // wavelength scaling for scattering
float l_k = pow(wva, -k);            // wavelength scaling for absorption
float ABl = A * exp(-B / wva);       // pure-water absorption contribution
float sca = be[j] * l_a / (1 - d.g);
float abs = (D * ba[j] + E) * l_k + ABl * (1 + 0.01 * td[j]);
```

---

## `wv.dat`

**Read at:** lines 447–461

**Format:** CDF of photon detection probability vs wavelength

| Column | Quantity |
|--------|----------|
| 0 | Cumulative probability (must start at exactly 0.0, end at exactly 1.0, strictly increasing) |
| 1 | Wavelength [nm], written with trailing period (e.g. `420.`) |

PPC samples a uniform random number and interpolates this table to pick the
photon wavelength. The shape encodes the DOM quantum efficiency × Cherenkov
photon spectrum weighting.

---

## `as.dat`

**Read at:** lines 330–343 (requires `ASENS` compile flag)

**Format:** up to `ANUM + 1` whitespace-separated floats on a single line.

The first value is a global angular sensitivity scale factor (`mas`). Subsequent
values are polynomial coefficients describing the angular acceptance of the DOM
as a function of photon incidence angle. A single value of `1` gives isotropic
acceptance.

---

## `tilt.dat` and `tilt.par`

**Read at:** lines 352–400 (requires `TILT` compile flag)

These encode the physical tilt of ice/dust layers inferred from dust-logger data
across multiple strings. `tilt.par` lists the horizontal distances (from a
reference string) of the dust-logger strings used. `tilt.dat` gives the
depth-dependent layer displacement at each of those distances.

For a homogeneous medium (water), the tilt is zero. The south-pole files can be
copied as-is; PPC uses them to correct photon propagation for tilted layer
boundaries and they have no effect if all entries are zero or the geometry is
uniform.

See: [Chirkin & Rongen (2013)](https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2009JD013741)

---

## `rnd.txt`

**Read at:** lines 405–415

Pre-computed table of unsigned 32-bit random multipliers for the GPU random
number generator. Do not modify; copy from an existing table directory.

---

# Water Optical Properties for PPC

## What is known

### Absorption length

| Site | λ [nm] | Absorption length [m] | Reference |
|------|--------|-----------------------|-----------|
| Mediterranean (ANTARES site, 2475 m) | 470 | 55–65 | Aguilar et al. 2005 (ANTARES) |
| Mediterranean (KM3NeT/ARCA, 3500 m) | 450 | ~60–70 | KM3NeT collaboration |
| Mediterranean (KM3NeT/ORCA, 2450 m) | 450 | ~50–60 | KM3NeT collaboration |
| Pacific (Cascadia Basin / P-ONE, 2660 m) | 450 | ~25–30 | STRAW, Boivin et al. 2021 |
| Lake Baikal (GVD site, ~1100 m) | 480 | ~20–25 | Baikal-GVD collaboration |
| Pure water (laboratory) | 420 | ~70–100 | Smith & Baker 1981 |

The Mediterranean is the most transparent natural water body currently
instrumented. Absorption is dominated by dissolved organic matter (CDOM) and
particles at wavelengths below ~450 nm, and by pure water above ~500 nm.

### Scattering

Effective scattering length b_e = b * (1 − g):

| Site | λ [nm] | b_e [m⁻¹] | Scattering length [m] | Reference |
|------|--------|-----------|----------------------|-----------|
| ANTARES site | 470 | ~0.005–0.01 | ~100–200 | ANTARES in-situ |
| Cascadia Basin | 450 | ~0.01 | ~100 | STRAW 2021 |
| Lake Baikal | 480 | ~0.003–0.006 | ~150–300 | Baikal |

### Scattering asymmetry parameter g

- Deep sea water: g ≈ 0.92–0.97 (dominated by Mie scattering from particles)
- Pure water (molecular): g ≈ 0 (Rayleigh)
- In practice the effective g for deep ocean water is ~0.94–0.96

### PMT quantum efficiency

QE curves are well characterised for the main photosensors in use:
- Hamamatsu R7081 (10" IceCube DOM): peak QE ~25% at ~390 nm
- Hamamatsu R12199 (3" KM3NeT PMT): peak QE ~28–32% at ~420 nm
- CATIROC/KM3NeT multi-PMT module: 31 × 3" PMTs per module
- XP53B22 (Baikal QUASAR-370): peak ~20% at ~370 nm

Manufacturer datasheets are publicly available; in-situ calibrations broadly
confirm them within ~10%.

---

## What is still needed / uncertain

### High priority — required for any realistic PPC water simulation

| Quantity | Status | Notes |
|----------|--------|-------|
| Absorption coefficient vs wavelength (300–600 nm) at target site | **Needed** | Must be measured in-situ; lab pure-water values are a lower bound |
| Effective scattering length vs wavelength at target site | **Needed** | Scattering is harder to measure than absorption; sparse data |
| g at target site and depth | **Uncertain** | Particle size distribution drives g; rarely measured directly in deep ocean |
| icemodel.par equivalents (α, κ, A, B) fit to water data | **Needed** | The SPICE model was fit to South Pole ice; a water-specific fit is required |

### Medium priority — important for accuracy

| Quantity | Status | Notes |
|----------|--------|-------|
| Depth dependence of absorption and scattering | **Sparse** | ANTARES observed ~10–15% variation with depth; deeper sites less characterised |
| Temporal / seasonal variation | **Known to exist** | ANTARES documented ~5–10% seasonal variation; amplitude site-dependent |
| Salinity and temperature corrections (D, E parameters) | **Unknown for water** | The SPICE D, E were fit to Antarctic ice; water has different temperature gradients |
| Absorption below 350 nm | **Poorly constrained** | Relevant for near-UV Cherenkov emission; CDOM absorbs strongly here |

### Lower priority — second-order effects

| Quantity | Status | Notes |
|----------|--------|-------|
| Angular scattering phase function (not just g) | **Unknown in situ** | SAM fraction `sf` in `cfg.txt` is unconstrained for water |
| Birefringence / anisotropy | **Not applicable** | Ice-specific; irrelevant for water |
| DOM angular acceptance in water | **Available** | QE vs angle characterised in lab; `as.dat` can be set from datasheet |
| Tilt corrections | **Not needed** | Water has no ice-layer tilt; `tilt.dat`/`tilt.par` can be zeroed |

---

# PPC Parameter Fits to Water Data

Fitted PPC `icemodel.par` equivalents from in-situ measurements using a
Monte Carlo uncertainty approach: each measurement range is sampled uniformly
50 000 times, the PPC model is fit to each sample, and the median and 16th/84th
percentiles of the resulting parameter distributions are reported as the central
value and 1σ interval.

## Model

**Absorption** (`icemodel.par` rows 3–4, applied in `ini.cxx:591`):

```
a(λ) = A · exp(−B/λ)    [dominant term; dust/CDOM neglected for clean deep water]
```

Note: B < 0 for water (UV-rising absorption), whereas B > 0 for ice (IR tail).
The dust power-law term `a_dust · (λ/400)^(−κ)` is set to zero here because
the Mediterranean and Pacific deep-water CDOM contribution is small at λ > 370 nm.
For Baikal it is non-negligible but cannot be separated from the exponential
with only two wavelength points.

**Effective scattering** (`icemodel.par` rows 1, `cfg.txt` row 4, `ini.cxx:591`):

```
b_e(λ) = b_e(400) · (λ/400)^(−α)
```

where b_e = b · (1 − g). g is fixed from the literature for each site.

## Results

### Mediterranean (KM3NeT/ANTARES)

Data: absorption at 375 nm (26 m) and 473 nm (60 m) from ANTARES; effective
scattering at 375 nm (122 m) and 473 nm (265 m) from ANTARES optical beacons.

| Parameter | Median | 1σ interval | Notes |
|-----------|--------|-------------|-------|
| A | 7.3 × 10⁻⁴ m⁻¹ | [4.0 × 10⁻⁴, 1.4 × 10⁻³] | Pure-water absorption amplitude |
| B | −1484 nm | [−1737, −1227] | Negative = UV rising (opposite sign to ice) |
| b_e(400) | 6.5 × 10⁻³ m⁻¹ | [5.9 × 10⁻³, 7.2 × 10⁻³] | Effective scattering at 400 nm |
| α | 3.24 | [2.54, 3.94] | Scattering wavelength slope |
| g | 0.92 (fixed) | — | Mie asymmetry; literature value |

The α ≈ 3.2 is consistent with the KM3NeT Rayleigh/Mie decomposition
(17% Rayleigh, 83% Mie): pure Rayleigh gives α ≈ 4.3, pure Mie gives α ≈ 1–2,
so a weighted combination landing near 3 is physically reasonable.

Consistency check (median fit vs data):

| λ [nm] | Quantity | Measured | Fit |
|--------|----------|----------|-----|
| 375 | Absorption length [m] | 26.0 | 26.2 |
| 473 | Absorption length [m] | 60.0 | 59.4 |
| 375 | Eff. scattering length [m] | 122 | 125 |
| 473 | Eff. scattering length [m] | 265 | 264 |

### Baikal-GVD

Data: absorption at 400 nm (8–12 m) and 488 nm (21–24 m) from ASP-15/Baikal-5D
instruments (arXiv:2309.16300). No multi-wavelength scattering data available.

| Parameter | Median | 1σ interval | Notes |
|-----------|--------|-------------|-------|
| A | 1.1 × 10⁻³ m⁻¹ | [5.7 × 10⁻⁴, 2.0 × 10⁻³] | Higher than Mediterranean; more CDOM |
| B | −1798 nm | [−2123, −1513] | |
| α | — | — | Insufficient scattering data |
| g | 0.88 (fixed) | — | Midpoint of reported 0.86–0.90 |

The larger A and more negative B compared to the Mediterranean reflect
the higher CDOM/particulate load in Lake Baikal.

### P-ONE/STRAW (Cascadia Basin)

Data: absorption at 450 nm (~30 m) from STRAW; effective scattering at 450 nm
derived from geometric scattering length 75 m with g = 0.92 (arXiv:2603.09495).
Only one wavelength point per quantity — α is not meaningfully constrained.

| Parameter | Median | 1σ interval | Notes |
|-----------|--------|-------------|-------|
| A | 2.3 × 10⁻³ m⁻¹ | [2.0 × 10⁻³, 2.5 × 10⁻³] | Tightly constrained by single point |
| B | −1206 nm | unconstrained | Cannot separate A from B with one point |
| b_e(400) | 2.9 × 10⁻³ m⁻¹ | [2.5 × 10⁻³, 3.8 × 10⁻³] | Long effective scattering length ~340 m |
| α | 9.1 (boundary) | unconstrained | Unphysical; single-wavelength fit hits upper bound |
| g | 0.92 (fixed) | — | |

A second scattering wavelength (e.g. from a 405 nm POCAM source) is needed
before α can be estimated for this site.

## Mapping to PPC config files

| PPC file | Parameter | Mediterranean value |
|----------|-----------|---------------------|
| `icemodel.par` row 1 | α (scattering slope) | 3.24 |
| `icemodel.par` row 2 | κ (absorption dust slope) | set to 0 (no dust) |
| `icemodel.par` row 3 | A (pure-water amplitude) | 7.3 × 10⁻⁴ |
| `icemodel.par` row 4 | B (pure-water scale [nm]) | −1484 |
| `icemodel.par` row 5 | D (temperature scattering) | 0 (unknown for water) |
| `icemodel.par` row 6 | E (temperature absorption) | 0 (unknown for water) |
| `cfg.txt` row 4 | g | 0.92 |
| `icemodel.dat` col 1 | b_e(400) [m⁻¹] | 6.5 × 10⁻³ (depth-uniform) |
| `icemodel.dat` col 2 | a_dust(400) [m⁻¹] | ~0 (set small, e.g. 1 × 10⁻⁴) |
| `icemodel.dat` col 3 | Δτ | 0 (no depth dependence) |
