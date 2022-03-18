class Constants(object):
    class CherenkovLightYield(object):
        """Constants relating to LY."""

        photons_per_GeV = 5.3 * 250 * 1e2

        class AngDist(object):
            a = 4.27033
            b = -6.02527
            c = 0.29887
            d = -0.00103

    class BaseConstants(object):
        c_vac = 2.99792458e8  # m/s
        e = 1.60217662e-19  # Coulomb
        h = 6.62607015e-34  # Js
