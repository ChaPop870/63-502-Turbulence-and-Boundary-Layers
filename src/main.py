from config import Constants
from kplanes import KPlane0500, KPlane0947
from plots import Plotter


kplane0500 = KPlane0500(Constants.r_temp_path, Constants.r_w_path)
kplane0947 = KPlane0947(Constants.r_temp_path, Constants.r_w_path)

plotter = Plotter(kplane0500, kplane0947)
# plotter.plot_horizontal_cross_section()
plotter.plot_joint_marginal_pdfs()
