import numpy as np

class Ellipsoid(object):

    def __init__(self, a, b):

        self._a = a
        self._b = b

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    def r(self, theta):
        return (self.a * self.b) / np.sqrt((self.a*np.sin(theta))**2 + (self.b*np.cos(theta))**2)

    def __repr__(self):
        s = f"{self.a}\n"
        s = s + f"{self.b}"
        return s
