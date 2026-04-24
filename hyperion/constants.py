class Constants(object):
    """Physical and simulation constants used across Hyperion."""

    class CherenkovLightYield(object):
        """
        Constants relating to light yield (LY).

        Attributes
        ----------
        photons_per_GeV : float
            Number of Cherenkov photons per GeV (approximate).
        AngDist : class
            Nested namespace with angular distribution parameters (a, b, c, d).
        """

        photons_per_GeV = 5.3 * 250 * 1e2

        class AngDist(object):
            """Angular distribution parameters.

            Attributes
            ----------
            a : float
                Parameter a in angular distribution.
            b : float
                Parameter b in angular distribution.
            c : float
                Parameter c in angular distribution.
            d : float
                Parameter d in angular distribution.
            """

            a = 4.27033
            b = -6.02527
            c = 0.29887
            d = -0.00103

    class BaseConstants(object):
        """Base physical constants used in calculations.

        Attributes
        ----------
        c_vac : float
            Speed of light in vacuum (m/s).
        e : float
            Elementary charge (C).
        h : float
            Planck constant (Js).
        """

        c_vac = 2.99792458e8  # m/s
        e = 1.60217662e-19  # Coulomb
        h = 6.62607015e-34  # Js
