## `as.dat`
None of these values are known

## `cfg.txt`
Configuration for `PPC`

- Oversizing of OM relative to nominal 16.32 cm
- OM officiency correction
- Fraction of the photon scattering angle that is accounted for by the simplified Liu scattering model. See section 5 of the [SPICE paper](https://user-web.icecube.wisc.edu/~dima/work/WISC/ppc/spice/paper/a.pdf) for further details.
- Expectation value of cos(theta) for the scattering models, where theta is the photon scattering angle. See section 5 of the [SPICE paper](https://user-web.icecube.wisc.edu/~dima/work/WISC/ppc/spice/paper/a.pdf) for further details.

## `icemodel.dat`
This gives the depth dependence of the scattering and absorption
First column is depth in meters
Second column is the absorption coeficient in m^{-1}
Third column is the scattering coefficient in m^{-1}
Fourth column is delta tau   (d) = T (d) − T (1730 m) with T = 221.5 − 0.00045319 · d + 5.822 · 10−6 · d

## `icemodel.par`

This gives the shape parameters of the ice model. These are referred to by various names, depending on if you look in [The SPICE paper](https://user-web.icecube.wisc.edu/~dima/work/WISC/ppc/spice/paper/a.pdf) or in the original [AMANDA optical properties paper](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2005JD006687). If they are referred to twice, then the first is the SPICE term, and the second is the AMANDA term

The first column is the value, and the second column is the uncertainty on that value. The uncertainty is not used by the code

- First value is the fit value of alpha that defines the wavelength dependence of the scattering
- Second value is the fit value of kappa that defines the wavelength dependence of the absorption
- Third term is A/A_{IR} the infrared stuff
- Fourth term is B/\lambda_{0}

## `rnd.txt`

This is used for random number generation. Probably don't change it

## `tilt.dat`
Dust logger data from 6 strings that give a sense of the tilt of the ice by matching the dust layers. See [this paper](https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2009JD013741) for more details

## `tilt.par`
Unknown but I think this is tied to the ice model. Probably do not change this.

## `wv.dat`
wavelength-tabulated DOM acceptance calculated from qe_dom2007a table of efficiency.h file of photonics
