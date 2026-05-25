from config import Constants
from kplanes import (KPlane0500, KPlane0947, KPlaneCombined)
from plots import Plotter


kplane0500 = KPlane0500(Constants.r_500_temp_path, Constants.r_500_w_path)
kplane0947 = KPlane0947(Constants.r_947_temp_path, Constants.r_947_w_path)
kplane_combined = KPlaneCombined(
    temp_path_500=Constants.r_500_temp_path,
    w_path_500=Constants.r_500_w_path,
    temp_path_947=Constants.r_947_temp_path,
    w_path_947=Constants.r_947_w_path
)


plotter = Plotter(kplane0500, kplane0947, kplane_combined)
# plotter.plot_horizontal_cross_section()
plotter.plot_joint_marginal_pdfs()
