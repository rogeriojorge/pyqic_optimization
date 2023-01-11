#!/usr/bin/env python3
import os
import numpy as np
from qic import Qic
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pathlib import Path
try: from optimized_configuration_nfp1 import optimized_configuration_nfp1
except: pass
try: from optimized_configuration_nfp2 import optimized_configuration_nfp2
except: pass
try: from optimized_configuration_nfp3 import optimized_configuration_nfp3
except: pass

this_path = Path(__file__).parent.resolve()

def plot_results(stel):
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(stel.varphi,stel.B2cQI,label='B2c')
    plt.plot(stel.varphi,stel.B2QI_factor,label='B2QI_factor')
    plt.legend();plt.xlabel(r'$\varphi$')
    plt.subplot(2, 2, 2)
    plt.plot(stel.varphi, stel.B20QI_deviation,label='B20 deviation')
    plt.plot(stel.varphi, stel.B2cQI_deviation,label='B2c deviation')
    plt.plot(stel.varphi, stel.B2sQI_deviation,label='B2s deviation')
    plt.legend();plt.xlabel(r'$\varphi$')
    plt.subplot(2, 2, 3)
    plt.plot(stel.varphi, stel.B20,label='B20')
    plt.legend();plt.xlabel(r'$\varphi$')
    plt.subplot(2, 2, 4)
    plt.plot(stel.varphi, stel.B2sQI,label='B2s')
    plt.legend();plt.xlabel(r'$\varphi$')
    plt.tight_layout()

def initial_configuration(nphi=131,order = 'r2',nfp=1):
    rc      = [ 1.0,0.0,-0.3,0.0,0.01,0.0,0.001 ]
    zs      = [ 0.0,0.0,-0.2,0.0,0.01,0.0,0.001 ]
    B0_vals = [ 1.0,0.1 ]
    omn_method ='non-zone-fourier'
    k_buffer = 1
    p_buffer = 2
    d_over_curvature = 0.5
    delta   = 0.1
    d_svals = [ 0.0,0.01,0.01,0.01 ]
    X2s_svals = [ 0.0,0.0,0.0,0.0 ]
    X2c_cvals = [ 0.0,0.0,0.0 ]
    X2s_cvals = [ 0.01,0.01,0.01 ]
    X2c_svals = [ 0.0,0.01,0.01,0.01 ]
    p2      =  0.0
    return Qic(omn_method = omn_method, p_buffer=p_buffer, p2=p2, delta=delta, k_buffer=k_buffer, rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, B2c_cvals=X2c_cvals, B2s_svals=X2s_svals, order=order, d_over_curvature=d_over_curvature, B2c_svals=X2c_svals, B2s_cvals=X2s_cvals)

def print_results(stel,initial_obj=0):
    out_txt  = f'from qic import Qic\n'
    out_txt += f'def optimized_configuration_nfp{stel.nfp}(nphi=131,order = "r2"):\n'
    out_txt += f'    rc      = [{",".join([str(elem) for elem in stel.rc])}]\n'
    out_txt += f'    zs      = [{",".join([str(elem) for elem in stel.zs])}]\n'
    out_txt += f'    B0_vals = [{",".join([str(elem) for elem in stel.B0_vals])}]\n'
    out_txt += f'    omn_method = "{stel.omn_method}"\n'
    out_txt += f'    k_buffer = {stel.k_buffer}\n'
    out_txt += f'    p_buffer = {stel.p_buffer}\n'
    out_txt += f'    d_over_curvature = {stel.d_over_curvature}\n'
    out_txt += f'    delta   = {stel.delta}\n'
    out_txt += f'    d_svals = [{",".join([str(elem) for elem in stel.d_svals])}]\n'
    out_txt += f'    nfp     = {stel.nfp}\n'
    out_txt += f'    iota    = {stel.iota}\n'
    if not stel.order=='r1':
        out_txt += f'    X2s_svals = [{",".join([str(elem) for elem in stel.B2s_svals])}]\n'
        out_txt += f'    X2c_cvals = [{",".join([str(elem) for elem in stel.B2c_cvals])}]\n'
        out_txt += f'    X2s_cvals = [{",".join([str(elem) for elem in stel.B2s_cvals])}]\n'
        out_txt += f'    X2c_svals = [{",".join([str(elem) for elem in stel.B2c_svals])}]\n'
        out_txt += f'    p2      = {stel.p2}\n'
        if not stel.p2==0:
            out_txt += f'    # DMerc mean  = {np.mean(stel.DMerc_times_r2)}\n'
            out_txt += f'    # DWell mean  = {np.mean(stel.DWell_times_r2)}\n'
            out_txt += f'    # DGeod mean  = {np.mean(stel.DGeod_times_r2)}\n'
        out_txt += f'    # B20QI_deviation_max = {stel.B20QI_deviation_max}\n'
        out_txt += f'    # B2cQI_deviation_max = {stel.B2cQI_deviation_max}\n'
        out_txt += f'    # B2sQI_deviation_max = {stel.B2sQI_deviation_max}\n'
        out_txt += f'    # Max |X20| = {max(abs(stel.X20))}\n'
        out_txt += f'    # Max |Y20| = {max(abs(stel.Y20))}\n'
        if stel.order == 'r3':
            out_txt += f'    # Max |X3c1| = {max(abs(stel.X3c1))}\n'
        out_txt += f'    # gradgradB inverse length: {stel.grad_grad_B_inverse_scale_length}\n'
        out_txt += f'    # d2_volume_d_psi2 = {stel.d2_volume_d_psi2}\n'
    out_txt += f'    # max curvature_d(0) = {stel.d_curvature_d_varphi_at_0}\n'
    out_txt += f'    # max d_d(0) = {stel.d_d_d_varphi_at_0}\n'
    out_txt += f'    # max gradB inverse length: {np.max(stel.inv_L_grad_B)}\n'
    out_txt += f'    # Max elongation = {stel.max_elongation}\n'
    if not initial_obj==0:
        out_txt += f'    # Initial objective = {initial_obj}\n'
    out_txt += f'    # Final objective = {obj(stel)}\n'
    out_txt += f'    return Qic(omn_method = omn_method, delta=delta, p_buffer=p_buffer, p2=p2, k_buffer=k_buffer, rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, B2c_cvals=X2c_cvals, B2s_svals=X2s_svals, order=order, d_over_curvature=d_over_curvature, B2c_svals=X2c_svals, B2s_cvals=X2s_cvals)'
    with open(os.path.join(this_path,f'optimized_configuration_nfp{stel.nfp}.py'), 'w') as f:
        f.write(out_txt)
    print(out_txt)
    return out_txt

def fun(dofs, stel, parameters_to_change, info={'Nfeval':0}, obj_array=[]):
    info['Nfeval'] += 1
    new_dofs = stel.get_dofs()
    for count, parameter in enumerate(parameters_to_change): new_dofs[stel.names.index(parameter)] = dofs[count]
    try:
        stel.set_dofs(new_dofs)
        objective_function = obj(stel)
        print(f"fun#{info['Nfeval']} -", f"\N{GREEK CAPITAL LETTER DELTA}B20 = {stel.B20QI_deviation_max:.2f},",
                                        f"\N{GREEK CAPITAL LETTER DELTA}B2c = {stel.B2cQI_deviation_max:.2f},",
                                        f"\N{GREEK CAPITAL LETTER DELTA}B2s = {stel.B2sQI_deviation_max:.2f},",
                                        f"1/L\N{GREEK CAPITAL LETTER DELTA}B = {np.max(stel.inv_L_grad_B):.2f},",
                                        f"1/L\N{GREEK CAPITAL LETTER DELTA}\N{GREEK CAPITAL LETTER DELTA}B = {stel.grad_grad_B_inverse_scale_length:.2f},",
                                        f"B0(1) = {stel.B0_vals[1]:.2f},",
                                        f"J = {objective_function:.2f}")
        obj_array.append(objective_function)
    except Exception as e:
        print(e)
        objective_function = 1e3
    return objective_function

def obj(stel):
    return np.sum(5*stel.B2cQI_deviation**2 + stel.B2sQI_deviation**2 + stel.B20QI_deviation**2 + stel.inv_L_grad_B**2 + stel.grad_grad_B_inverse_scale_length_vs_varphi**2 + 0.05*stel.elongation**2)/stel.nphi + 50*stel.B0_vals[1]**2

def main():
    # stel = initial_configuration(nfp=3)
    # stel = optimized_configuration_nfp1(151)
    # stel = optimized_configuration_nfp2(151)
    stel = optimized_configuration_nfp3(151)
    # stel.plot_boundary(r=0.05)
    # exit()

    initial_obj = obj(stel)
    initial_dofs = stel.get_dofs()
    parameters_to_change = (['zs(2)','B0(1)','ds(1)','B2cs(1)','B2sc(0)','d_over_curvature',
                             'zs(4)','rc(2)','ds(2)','B2cs(2)','B2sc(1)',
                             'zs(6)','rc(4)','ds(3)','B2cs(3)','B2sc(2)'])
    dofs = [initial_dofs[stel.names.index(parameter)] for parameter in parameters_to_change]

    plot_results(stel)

    obj_array = []
    method = 'Nelder-Mead'
    maxiter = 3000
    maxfev  = maxiter
    res = minimize(fun, dofs, args=(stel, parameters_to_change, {'Nfeval':0}, obj_array), method=method, tol=1e-3, options={'maxiter': maxiter, 'maxfev': maxfev, 'disp': True})
    print_results(stel, initial_obj)

    plot_results(stel)

    plt.figure();plt.xlabel('Function evaluation')
    plt.plot(np.array(obj_array));plt.ylabel('Objective function')

    stel.B_contour(show=False)
    plt.show()

    # stel.plot_boundary()

if __name__ == "__main__":
    main()