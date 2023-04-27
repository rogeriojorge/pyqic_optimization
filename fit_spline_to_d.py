import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy import optimize, interpolate
from qic import Qic

# Points to fit with
N_points = 7

# Original configuration
stel = Qic.from_paper("QI NFP1 r2")
d_over_curvature = stel.d/stel.curvature

# Construct interpolation for that data
x_in = stel.phi
y_in = d_over_curvature
x_in_periodic = np.append(x_in, 2*np.pi/stel.nfp)
y_in_periodic = np.append(y_in, y_in[0])
spline_d_over_curv = interpolate.make_interp_spline(x_in_periodic, y_in_periodic, 
                                                    bc_type = 'periodic', k = 7)  # Use the same form as in the code (just for consistency)

# Now define the control points as in the code, and evaluate the function at those
x_new = np.linspace(0,1,N_points)*np.pi/stel.nfp
x_new_periodic = np.append(x_new, 2*np.pi/stel.nfp-x_new[-2::-1])
y_new = spline_d_over_curv(x_new)
# y_new constitute the new control points
print('d_spline coefficients: ', y_new)

# Compare the d from the new input to the original
y_new_periodic = np.append(y_new, y_new[-2::-1])
spline_d_over_curv_new = interpolate.make_interp_spline(x_new_periodic, y_new_periodic)
temp_d_over_curv = spline_d_over_curv_new(stel.phi)
plt.plot(stel.phi, stel.d/stel.curvature)
plt.plot(stel.phi, temp_d_over_curv, linestyle = '--')
plt.show()

# Construct a stel to first order with the new spline input
properties = ["nphi", "nfp", "rc", "zs", "B0_vals", "omn_method", "k_buffer", "p_buffer", "delta", "omn"]
vals = {}
for prop in properties:
    vals[prop] = getattr(stel,prop)
vals["order"] = 'r1'
d_over_curvature_spline = y_new

stel_new = Qic(omn_method = vals["omn_method"], delta=vals["delta"], p_buffer=vals["p_buffer"], 
           k_buffer=vals["k_buffer"], rc=vals["rc"], zs=vals["zs"], nfp=vals["nfp"], 
           B0_vals=vals["B0_vals"], nphi=vals["nphi"], omn=vals["omn"], order=vals["order"], 
           d_over_curvature_spline=d_over_curvature_spline) 
