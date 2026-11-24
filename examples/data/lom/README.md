# Published LOM data

Used by `examples/06_fit_lom_shape.py` (fit) and `scripts/check_lom16_yield.py` (check). All sources are public.

| File | Contents | Source |
|------|----------|--------|
| `lom16_zenith_400nm.txt`, `lom18_zenith_400nm.txt` | cos(zenith), effective area [cm^2] at 400 nm | [1], Fig. 75 left |
| `lom16_wavelength.txt`, `lom18_wavelength.txt` | wavelength [nm], direction-averaged effective area [cm^2] | [1], Fig. 75 right |
| `gen1_wavelength.txt` | wavelength [nm], direction-averaged effective area [cm^2] of the IceCube DOM | [1], Fig. 75 right |
| `lom16_relative_yield.txt`, `lom18_relative_yield.txt` | propagation distance [m], Cherenkov-weighted effective area over that of the IceCube DOM | [1], Fig. 76 |
| `lom16_start.conf` | PMT layout, tilts and vessel size in `om.conf` format | [2] |

The form of the PMT response is from [3].

## How they were made

**Fig. 75.** The figure is vector graphics. Page 125 of the PDF (printed page 119) was
converted with `pdftocairo -svg`, and the line coordinates were mapped to data using the
axis ticks. No smoothing was applied. Wavelengths were rounded to the 20 nm grid they
lie on (offsets below 0.02 nm) and areas below 0.005 cm^2 set to zero.

- [1] describes the curves as Geant4 results using measured PMT efficiencies, with the
  quantum efficiency of NNVT 4-inch PMTs. The right panel is averaged over solid angle.
  The text does not say whether the left panel is averaged over azimuth; the fit assumes so.
- The area rises from 640 to 680 nm in both wavelength curves. That is in the figure, is
  not expected of a PMT, and is below 0.6 cm^2.

**Fig. 76.** Page 126 of the PDF (printed page 120), converted in the same way.

- The ice model used is not stated near the figure. The horizontal axis is
  the photon path length, so the curve depends on the ice.

**`lom16_start.conf`.** From the constants in `OMSimLOM16.hh` [2]: ring tilts 36 and 62 deg,
ring azimuths offset by 45 deg. `Rr` is the glass radius plus the bulge of the cylinder,
153.2 + 68.8 sin(2.5 deg) = 156.2 mm. `Rz` is 153.2 + 68.8 = 222.0 mm. Only these numbers
were taken, not code. `beta = 1` is an assumption. 171 is the type ID.

## References

1. IceCube-Gen2 Collaboration, *The IceCube-Gen2 Neutrino Observatory*, Technical Design
   Report, Parts I and II, version 0.05, 27 July 2023.
   <https://icecube-gen2.wisc.edu/wp-content/uploads/2023/07/IceCube_Gen2_TDR_0.05_July27.2023_Part_I_and_II.pdf>
   (SHA-256 `f573b56307a6627447d0a1805d2d5b3126aae49a6f9622eb7d7a7ce787cb4e5d`)
2. OMSim, `common/geometry_construction/include/OMSimLOM16.hh`, commit
   `dc9c247cdfe450be486e7d75e51856d508bebb0a`. <https://github.com/icecube/OMSim>
3. PPC V1.0.1, `gpu/ini.cxx`, function `itype::f`. <https://github.com/icecube/ppc>
