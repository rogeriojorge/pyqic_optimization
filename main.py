#!/usr/bin/env python3
import numpy as np
from qic import Qic
import unicodedata as ud
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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

def initial_configuration(nphi=131,order = 'r2'):
    rc      = [ 1.0,0.0,-0.2,0.0,-0.01,0.0,0.001]
    zs      = [ 0.0,0.0,-0.2,0.0,-0.01,0.0,0.001]
    sigma0  =  0.0
    B0_vals = [ 1.0,0.1]
    omn_method ='non-zone'
    k_buffer = 3
    d_over_curvature   = 0.5
    d_svals = [ 0.0,0.01,0.01,0.01]
    nfp     = 1
    B2s_svals = [ 0.0,0.2,0.01,0.01]
    B2c_cvals = [ 0.2,0.01,0.01]
    B2c_svals = [ 0.0,0.2,0.01,0.01]
    B2s_cvals = [ 0.2,0.01,0.01]
    return Qic(omn_method = omn_method, k_buffer=k_buffer, rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, B2c_cvals=B2c_cvals, B2s_svals=B2s_svals, order=order, d_over_curvature=d_over_curvature, B2c_svals=B2c_svals, B2s_cvals=B2s_cvals)

def optimized_configuration(nphi=131,order = 'r2'):
    rc      = [ 1.0,0.0,-0.19723768011222192,0.0,-0.0033985927586399206,0.0,0.001188229120486172 ]
    zs      = [ 0.0,0.0,-0.1443012447143185,0.0,-0.004440655196058562,0.0,-0.00011745921347312632 ]
    B0_vals = [ 1.0,0.20451082571596524 ]
    omn_method ='non-zone'
    k_buffer = 3
    d_over_curvature   = 0.6503556225285828
    d_svals = [ 0.0,0.0021453885350019948,0.006421068459924151,0.019415598371449034 ]
    nfp     = 1
    iota    =  -0.5053918535299964
    B2s_svals = [ 0.0,0.06419657126582455,0.005305886379085124,0.01542637793760189 ]
    B2c_cvals = [ 0.24082770898419342,0.017178206256682788,0.00211109130815362 ]
    B2s_cvals = [ 0.19220962640346062,0.015900291094018353,0.015132252008301327 ]
    B2c_svals = [ 0.0,0.5084386277822914,0.012258988846692458,0.006842184458937944 ]
    p2      =  0.0
    # B20QI_deviation_max = 0.7724472436695234
    # B2cQI_deviation_max = 1.5878762401955058
    # B2sQI_deviation_max = 1.2946414542314466
    # Max |X20| = 0.6089023938728432
    # Max |Y20| = 2.5813821514861703
    # gradgradB inverse length: 3.720599423080629
    # d2_volume_d_psi2 = 228.53167239983208
    # max curvature'(0): 2.647115604886152
    # max d'(0): 1.7947457196006908
    # max gradB inverse length: 2.070297376962882
    # Max elongation = 5.511170121630197
    # Initial objective = 47.57231770408526
    # Final objective = 34.810129867834
    return Qic(omn_method = omn_method, k_buffer=k_buffer, rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, B2c_cvals=B2c_cvals, B2s_svals=B2s_svals, order=order, d_over_curvature=d_over_curvature, B2c_svals=B2c_svals, B2s_cvals=B2s_cvals)

def print_results(stel,initial_obj=0):
    print('    rc      = [',','.join([str(elem) for elem in stel.rc]),']')
    print('    zs      = [',','.join([str(elem) for elem in stel.zs]),']')
    print('    B0_vals = [',','.join([str(elem) for elem in stel.B0_vals]),']')
    print("        omn_method ='"+stel.omn_method+"'")
    print("        k_buffer =",stel.k_buffer)
    print('    d_over_curvature   =',stel.d_over_curvature)
    print('    d_svals = [',','.join([str(elem) for elem in stel.d_svals]),']')
    print('    nfp     =',stel.nfp)
    print('    iota    = ',stel.iota)
    if not stel.order=='r1':
        print('    B2s_svals = [',','.join([str(elem) for elem in stel.B2s_svals]),']')
        print('    B2c_cvals = [',','.join([str(elem) for elem in stel.B2c_cvals]),']')
        print('    B2s_cvals = [',','.join([str(elem) for elem in stel.B2s_cvals]),']')
        print('    B2c_svals = [',','.join([str(elem) for elem in stel.B2c_svals]),']')
        print('    p2      = ',stel.p2)
        if not stel.p2==0:
            print('    # DMerc mean  =',np.mean(stel.DMerc_times_r2))
            print('    # DWell mean  =',np.mean(stel.DWell_times_r2))
            print('    # DGeod mean  =',np.mean(stel.DGeod_times_r2))
        # print('    # B20 mean =',np.mean(stel.B20))
        print('    # B20QI_deviation_max =',stel.B20QI_deviation_max)
        print('    # B2cQI_deviation_max =',stel.B2cQI_deviation_max)
        print('    # B2sQI_deviation_max =',stel.B2sQI_deviation_max)
        print('    # Max |X20| =',max(abs(stel.X20)))
        print('    # Max |Y20| =',max(abs(stel.Y20)))
        if stel.order == 'r3':
            print('    # Max |X3c1| =',max(abs(stel.X3c1)))
        print('    # gradgradB inverse length:', stel.grad_grad_B_inverse_scale_length)
        print('    # d2_volume_d_psi2 =',stel.d2_volume_d_psi2)
    print("        # max curvature'(0):", stel.d_curvature_d_varphi_at_0)
    print("        # max d'(0):", stel.d_d_d_varphi_at_0)
    print('    # max gradB inverse length:', np.max(stel.inv_L_grad_B))
    print('    # Max elongation =',stel.max_elongation)
    if not initial_obj==0:
        print('    # Initial objective =',initial_obj)
    print('    # Final objective =',obj(stel))

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
                                        f"J = {objective_function:.2f}")
        obj_array.append(objective_function)
    except Exception as e:
        print(e)
        objective_function = 1e3
    return objective_function

def obj(stel):
    return 10.0*stel.B2cQI_deviation_max**2 + stel.B2sQI_deviation_max**2 + stel.B20QI_deviation_max**2 + np.max(stel.inv_L_grad_B)**2 + 0.01*stel.B0_vals[1]**2 + 0.1*stel.max_elongation**2

def main():
    # stel = initial_configuration()
    stel = optimized_configuration(151)
    initial_obj = obj(stel)

    parameters_to_change = (['zs(2)','B0(1)','ds(1)','B2ss(1)','B2cc(0)','B2cs(1)','B2sc(0)','d_over_curvature',
                             'zs(4)','rc(2)','ds(2)','B2ss(2)','B2cc(1)','B2cs(2)','B2sc(1)',
                             'zs(6)','rc(4)','ds(3)','B2ss(3)','B2cc(2)','B2cs(3)','B2sc(2)'])
    initial_dofs = stel.get_dofs()
    dofs = [initial_dofs[stel.names.index(parameter)] for parameter in parameters_to_change]

    plot_results(stel)

    obj_array = []
    method = 'Nelder-Mead'
    maxiter = 2000
    maxfev  = 2000
    res = minimize(fun, dofs, args=(stel, parameters_to_change, {'Nfeval':0}, obj_array), method=method, tol=1e-3, options={'maxiter': maxiter, 'maxfev': maxfev, 'disp': True})
    print_results(stel, initial_obj)

    plot_results(stel)

    plt.figure();plt.xlabel('Function evaluation')
    # plt.plot(np.log(np.array(obj_array)));plt.ylabel('Log(Objective function)')
    plt.plot(np.array(obj_array));plt.ylabel('Objective function')

    # stel.plot_boundary()
    stel.B_contour(show=False)

    plt.show()

if __name__ == "__main__":
    main()