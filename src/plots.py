import matplotlib.pyplot as plt

from kplanes import (KPlane0500, KPlane0947)


class Plotter:

    def __init__(self, kplane0500: KPlane0500, kplane0947: KPlane0947) -> None:
        self.kplane0500 = kplane0500
        self.kplane0947 = kplane0947

    def plot_horizontal_cross_section(self):
        ...