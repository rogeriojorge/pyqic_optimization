#!/usr/bin/env python3
import os
import time
import shutil
import argparse
import vmecPlot2
import subprocess
import numpy as np
from qic import Qic
import booz_xform as bx
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import interpolate
from qic.calculate_r2 import evaluate_X2c_X2s_QI
from simsopt.mhd import Vmec, Boozer
from neat.fields import Simple
from neat.tracing import ChargedParticleEnsemble, ParticleEnsembleOrbit_Simple
try: from optimized_configuration_nfp1 import optimized_configuration_nfp1
except: pass
try: from optimized_configuration_nfp2 import optimized_configuration_nfp2
except: pass
try: from optimized_configuration_nfp3 import optimized_configuration_nfp3
except: pass
this_path = Path(__file__).parent.resolve()

def plot_results(stel, show=False):
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(stel.varphi,stel.B2cQI,label='B2c')
    plt.plot(stel.varphi,stel.B2cQI_factor,label='B2QI_factor')
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
    if show: plt.show()

def initial_configuration(nphi=131,order = 'r3',nfp=1, N_d_over_curvature_spline=6):
    # rc      = [ 1.0,0.0,-0.41599809655680886,0.0,0.08291443961920232,0.0,-0.008906891641686355,0.0,0.0,0.0,0.0,0.0,0.0 ]
    # zs      = [ 0.0,0.0,-0.28721210154364263,0.0,0.08425262593215394,0.0,-0.010427621520053335,0.0,-0.0008921610906627226,0.0,-6.357200965811029e-07,0.0,2.7316247301500753e-07 ]
    # B0_vals = [1.0,0.18]
    # omn_method ='non-zone-fourier'
    # k_buffer = 1
    # p_buffer = 2
    # delta = 0.1
    # d_over_curvature_spline = [0.5]*N_d_over_curvature_spline
    # X2s_cvals = [0.001]*nphi
    # X2c_svals = [0.001]*nphi
    if nfp==1: stel_initial = Qic.from_paper('QI NFP1 r2', nphi=nphi)
    else: stel_initial = Qic.from_paper('QI NFP2 r2', nphi=nphi)
    d_over_curvature_cvals = []
    min_geo_qi_consistency = stel.min_geo_qi_consistency(order = 1)
    X2c, X2s = evaluate_X2c_X2s_QI(stel, X2s_in=0)
    d_over_curvature = stel_initial.d/stel_initial.curvature
    # Construct interpolation for that data
    x_in = stel_initial.phi
    y_in = d_over_curvature
    x_in_periodic = np.append(x_in, 2*np.pi/stel_initial.nfp)
    y_in_periodic = np.append(y_in, y_in[0])
    spline_d_over_curv = interpolate.make_interp_spline(x_in_periodic, y_in_periodic, 
                                                        bc_type = 'periodic', k = N_d_over_curvature_spline)  # Use the same form as in the code (just for consistency)
    # Now define the control points as in the code, and evaluate the function at those
    x_new = np.linspace(0,1,N_d_over_curvature_spline)*np.pi/stel_initial.nfp
    y_new = spline_d_over_curv(x_new)
    return Qic(omn_method = stel_initial.omn_method, delta=stel_initial.delta,
               p_buffer=stel_initial.p_buffer, k_buffer=stel_initial.k_buffer,
               rc=stel_initial.rc,zs=stel_initial.zs, nfp=stel_initial.nfp, B0_vals=stel_initial.B0_vals,
               nphi=stel.nphi, omn=True, order=order, d_over_curvature_cvals=d_over_curvature_cvals,
               B2c_svals=X2c, B2s_cvals=X2s, d_over_curvature_spline=y_new)

def print_results(stel,initial_obj=0, Print=True):
    out_txt  = f'from qic import Qic\n'
    out_txt += f'from qic.calculate_r2 import evaluate_X2c_X2s_QI\n'
    out_txt += f'def optimized_configuration_nfp{stel.nfp}(nphi={stel.nphi},order = "r3"):\n'
    out_txt += f'    rc      = [{",".join([str(elem) for elem in stel.rc])}]\n'
    out_txt += f'    zs      = [{",".join([str(elem) for elem in stel.zs])}]\n'
    out_txt += f'    B0_vals = [{",".join([str(elem) for elem in stel.B0_vals])}]\n'
    out_txt += f'    omn_method = "{stel.omn_method}"\n'
    out_txt += f'    k_buffer = {stel.k_buffer}\n'
    out_txt += f'    p_buffer = {stel.p_buffer}\n'
    out_txt += f'    d_over_curvature_cvals = [{",".join([str(elem) for elem in stel.d_over_curvature_cvals])}]\n'
    out_txt += f'    d_over_curvature_spline = [{",".join([str(elem) for elem in stel.d_over_curvature_spline])}]\n'
    out_txt += f'    delta   = {stel.delta}\n'
    out_txt += f'    d_svals = [{",".join([str(elem) for elem in stel.d_svals])}]\n'
    out_txt += f'    nfp     = {stel.nfp}\n'
    out_txt += f'    iota    = {stel.iota}\n'
    out_txt += f'    p2    = {stel.p2}\n'
    if not stel.order=='r1':
        out_txt += f'    X2s_svals = [{",".join([str(elem) for elem in stel.B2s_svals])}]\n'
        out_txt += f'    X2c_cvals = [{",".join([str(elem) for elem in stel.B2c_cvals])}]\n'
        out_txt += f'    X2s_cvals = [{",".join([str(elem) for elem in stel.B2s_cvals])}]\n'
        out_txt += f'    X2c_svals = [{",".join([str(elem) for elem in stel.B2c_svals])}]\n'
        if not stel.p2==0:
            out_txt += f'    # DMerc mean  = {np.mean(stel.DMerc_times_r2)}\n'
            out_txt += f'    # DWell mean  = {np.mean(stel.DWell_times_r2)}\n'
            out_txt += f'    # DGeod mean  = {np.mean(stel.DGeod_times_r2)}\n'
        out_txt += f'    # B20QI_deviation_max = {stel.B20QI_deviation_max}\n'
        out_txt += f'    # B2cQI_deviation_max = {stel.B2cQI_deviation_max}\n'
        out_txt += f'    # B2sQI_deviation_max = {stel.B2sQI_deviation_max}\n'
        out_txt += f'    # Max |X20| = {max(abs(stel.X20))}\n'
        out_txt += f'    # Max |Y20| = {max(abs(stel.Y20))}\n'
        out_txt += f'    # Max |Y2c| = {max(abs(stel.Y2c))}\n'
        out_txt += f'    # Max |Y2s| = {max(abs(stel.Y2s))}\n'
        out_txt += f'    # Min r_singularity = {np.min(stel.r_singularity)}\n'
        if stel.order == 'r3':
            out_txt += f'    # Max |X3c1| = {max(abs(stel.X3c1))}\n'
            out_txt += f'    # Max |X3s1| = {max(abs(stel.X3s1))}\n'
            out_txt += f'    # Max |Y3c1| = {max(abs(stel.Y3c1))}\n'
            out_txt += f'    # Max |Y3s1| = {max(abs(stel.Y3s1))}\n'
        out_txt += f'    # gradgradB inverse length: {stel.grad_grad_B_inverse_scale_length}\n'
        out_txt += f'    # d2_volume_d_psi2 = {stel.d2_volume_d_psi2}\n'
    out_txt += f'    # max curvature_d(0) = {stel.d_curvature_d_varphi_at_0}\n'
    out_txt += f'    # max d_d(0) = {stel.d_d_d_varphi_at_0}\n'
    out_txt += f'    # max gradB inverse length: {np.max(stel.inv_L_grad_B)}\n'
    out_txt += f'    # Max elongation = {stel.max_elongation}\n'
    if not initial_obj==0:
        out_txt += f'    # Initial objective = {initial_obj}\n'
    out_txt += f'    # Final objective = {obj_r3(stel)}\n'
    out_txt += f'    stel = Qic(omn_method = omn_method, delta=delta, p_buffer=p_buffer, p2=p2, k_buffer=k_buffer, rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, B2c_cvals=X2c_cvals, B2s_svals=X2s_svals, order=order, d_over_curvature_cvals=d_over_curvature_cvals, B2c_svals=X2c_svals, B2s_cvals=X2s_cvals, d_over_curvature_spline=d_over_curvature_spline)\n'
    out_txt += f'    stel._set_names();stel.calculate()\n'
    out_txt += f'    min_geo_qi_consistency = stel.min_geo_qi_consistency(order = 1);X2c, X2s = evaluate_X2c_X2s_QI(stel, X2s_in=0)\n'
    out_txt += f'    stel.B2c_svals=X2c;stel.B2s_cvals=X2s;stel._set_names();stel.calculate()\n'
    out_txt += f'    return stel'
    with open(os.path.join(this_path,f'optimized_configuration_nfp{stel.nfp}.py'), 'w') as f:
        f.write(out_txt)
    if Print: print(out_txt)
    return out_txt

def fun_r1(dofs, stel, parameters_to_change, info={'Nfeval':0}, obj_array=[], start_time = 0, B01=0.18):
    info['Nfeval'] += 1
    new_dofs = stel.get_dofs()
    for count, parameter in enumerate(parameters_to_change):
        new_dofs[stel.names.index(parameter)] = dofs[count]
    stel.set_dofs(new_dofs)
    stel._set_names()
    stel.calculate()
    min_geo_qi_consistency = stel.min_geo_qi_consistency(order = 1)
    X2c, X2s = evaluate_X2c_X2s_QI(stel, X2s_in=0)
    objective_function = obj_r1(stel, B0_well_depth=B01, min_geo_qi_consistency=min_geo_qi_consistency, X2c=X2c, X2s=X2s)
    print(f"fun#{info['Nfeval']} r1 -",
          f"min_geo_qi_consistency = {min_geo_qi_consistency:1f}, "
        + f"max(X2c) = {np.max(X2c):1f}, max(X2s) = {np.max(X2s):1f}, "
        + f"B0(1) = {stel.B0_vals[1]:.2f}, "
        + f"max(elong) = {np.max(stel.elongation):.2f}, "
        + f"J = {objective_function:.2f}")
    obj_array.append(objective_function)
    return objective_function

def fun_r3(dofs, stel, parameters_to_change, info={'Nfeval':0}, obj_array=[], start_time = 0, B01=0.18):
    info['Nfeval'] += 1
    new_dofs = stel.get_dofs()
    for count, parameter in enumerate(parameters_to_change):
        new_dofs[stel.names.index(parameter)] = dofs[count]
    stel.set_dofs(new_dofs)
    # stel.order='r1' # Why does evaluate_X2c_X2s_QI give different results depending on the order of the stel?
    stel._set_names();stel.calculate()
    min_geo_qi_consistency = stel.min_geo_qi_consistency(order = 1)
    X2c, X2s = evaluate_X2c_X2s_QI(stel, X2s_in=0)
    # stel.order='r3'
    stel.B2c_svals=X2c;stel.B2s_cvals=X2s
    stel._set_names();stel.calculate()
    objective_function = obj_r3(stel, B0_well_depth=B01)
    print(f"fun#{info['Nfeval']} r3 -",
        f"\N{GREEK CAPITAL LETTER DELTA}B2c = {stel.B2cQI_deviation_max:.2f},",
        f"1/L\N{GREEK CAPITAL LETTER DELTA}B = {np.max(stel.inv_L_grad_B):.2f},",
        f"1/L\N{GREEK CAPITAL LETTER DELTA}\N{GREEK CAPITAL LETTER DELTA}B = {stel.grad_grad_B_inverse_scale_length:.2f},",
        f"B0(1) = {stel.B0_vals[1]:.2f},",
        f"max(elong) = {stel.max_elongation:.2f},",
        f"min(r_singularity) = {np.min(stel.r_singularity):.2f},",
        f"max(X20,Y20) = {max(abs(stel.X20)):.2f},{max(abs(stel.Y20)):.2f},",
        f"max(X3s1,Y3c1) = {max(abs(stel.X3s1)):.2f},{max(abs(stel.Y3c1)):.2f},",
        f"\N{GREEK CAPITAL LETTER DELTA}\N{GREEK SMALL LETTER ALPHA} = {max(abs(stel.alpha - stel.alpha_no_buffer)):.2f},",
        f"J = {objective_function:.2f}")
    obj_array.append(objective_function)
    n_plot = 51
    n_save_results=2011
    time_in_seconds = int(time.time()-start_time)
    hours = time_in_seconds // 3600
    minutes = (time_in_seconds % 3600) // 60
    seconds = time_in_seconds % 60
    if np.mod(info['Nfeval'],n_plot)==0:
        log_obj_array = np.log(obj_array)
        # plt.clf()
        plt.plot(np.linspace(info['Nfeval']-n_plot,info['Nfeval'],n_plot),log_obj_array[-n_plot:],'b-')
        plt.gca().set_ylim(top=1.03*log_obj_array[0])
        # plt.gca().set_ylim(top=9)
        plt.gca().set_ylim(bottom=np.min(log_obj_array))
        plt.title(f'Running time: {hours:02d}h:{minutes:02d}m:{seconds:02d}s, minimum f = {np.min(obj_array):.2f}')
        plt.pause(1e-5)
    if np.mod(info['Nfeval'],n_save_results)==0:
        print_results(stel, 0, Print=False)
    return objective_function

def obj_r1(stel, min_geo_qi_consistency, X2c, X2s, B0_well_depth=0.18):
    return (min_geo_qi_consistency
           + 5e+1*(stel.B0_vals[1]-B0_well_depth)**2
           + 1e-2*np.max(X2c) + 1e-2*np.max(X2s)
           + 4e-2*np.max(stel.elongation)
    )

def obj_r3(stel, B0_well_depth=0.18):
    weight_B0vals = 3e3
    weight_d_at_0 = 1e-4
    weight_gradB_scale_length = 5e-6
    weight_elongation = 2e-2
    weight_d = 5e-5
    weight_alpha_diff = 1e-5
    weight_gradgradB_scale_length = 5e-4
    weight_min_geo_qi_consistency = 1e2
    weight_B20cs = 5e-6
    weight_B2c_dev = 1
    weight_XYZ3 = 1e-2
    weight_XYZ2 = 1e-2
    weight_r_singularity = 2e-2
    return (
         + weight_B0vals*(stel.B0_vals[1]-B0_well_depth)**2 
         + weight_gradB_scale_length*np.sum(stel.inv_L_grad_B**2)
         # + weight_gradgradB_scale_length*np.sum(stel.grad_grad_B_inverse_scale_length_vs_varphi**2)/stel.nphi
         + weight_r_singularity*np.min(stel.r_singularity)**(-2)
         + weight_elongation*np.sum(stel.elongation**2)/stel.nphi
         + weight_elongation*np.max(stel.elongation)**2
         + weight_B2c_dev*np.sum(stel.B2cQI_deviation**2 + stel.B2sQI_deviation_max**2 + stel.B20QI_deviation**2)/stel.nphi
         + weight_min_geo_qi_consistency*stel.min_geo_qi_consistency(order = 1)
         + weight_d*np.sum(stel.d**2)/stel.nphi
         + weight_d_at_0*stel.d_curvature_d_varphi_at_0**2
         + weight_d_at_0*stel.d_d_d_varphi_at_0**2
         + weight_B20cs*np.sum(stel.B20**2 + stel.B2c**2 + stel.B2s**2)/stel.nphi
         + weight_XYZ2*(np.max(stel.X20)+np.max(stel.X2c)+np.max(stel.X2s)
                       +np.max(stel.Y20)+np.max(stel.Y2c)+np.max(stel.Y2s)
                       +np.max(stel.Z20)+np.max(stel.Z2c)+np.max(stel.Z2s))**2
         + weight_XYZ3*(np.max(stel.X3c1)+np.max(stel.X3s1)
                      +np.max(stel.Y3c1)+np.max(stel.Y3s1))**2
         # + weight_alpha_diff*np.sum((stel.alpha - stel.alpha_no_buffer)**2)/stel.nphi
    )

def main(nfp=1, refine_optimization=False, nphi=91, maxiter = 3000, B01=0.18, N_d_over_curvature_spline=6, OUT_DIR=os.path.join(this_path,f'qic_nfp1'), show=True):
    if nfp not in [1,2,3]: raise ValueError('nfp should be 1, 2 or 3.')
    if refine_optimization:
        if   nfp==1: stel = optimized_configuration_nfp1(nphi)
        elif nfp==2: stel = optimized_configuration_nfp2(nphi)
        elif nfp==3: stel = optimized_configuration_nfp3(nphi)
    else: stel = initial_configuration(nfp=nfp, nphi=nphi, N_d_over_curvature_spline=N_d_over_curvature_spline)
    start_time = time.time()
    initial_obj = obj_r3(stel)
    print('Initial objective = ', initial_obj)
    stel.order = 'r1';stel._set_names();stel.calculate()
    ### Define degrees of freedom
    parameters_to_change = (['B0(1)','rc(2)','zs(2)','rc(4)','zs(4)','zs(6)','zs(8)','zs(10)','zs(12)'])
    [parameters_to_change.append(f'd_over_curvature_spline({i})') for i in range(len(stel.d_over_curvature_spline))]
    dofs = [stel.get_dofs()[stel.names.index(parameter)] for parameter in parameters_to_change]
    ### Start with r1 optimization for good initial guess
    obj_array = []
    plt.ion();fig, ax = plt.subplots()
    res = minimize(fun_r1, dofs, args=(stel, parameters_to_change, {'Nfeval':0}, obj_array, start_time, B01), method='Nelder-Mead', tol=1e-4, options={'maxiter': maxiter, 'maxfev': maxiter, 'disp': True})
    plt.savefig(os.path.join(OUT_DIR,f'order1_opt_nfp{stel.nfp}.pdf'));plt.close()
    ### Do r3 optimization
    stel.order='r3'
    obj_array = []
    dofs = [stel.get_dofs()[stel.names.index(parameter)] for parameter in parameters_to_change]
    plt.ion();fig, ax = plt.subplots()
    res = minimize(fun_r3, dofs, args=(stel, parameters_to_change, {'Nfeval':0}, obj_array, start_time, B01), method='Nelder-Mead', tol=1e-4, options={'maxiter': maxiter, 'maxfev': maxiter, 'disp': True})
    # res = minimize(fun_r3, dofs, args=(stel, parameters_to_change, {'Nfeval':0}, obj_array, start_time, B01), method='BFGS', tol=1e-4, options={'maxiter': maxiter/20, 'maxfev': maxfev/20, 'disp': True})
    plt.savefig(os.path.join(OUT_DIR,f'order3_opt_nfp{stel.nfp}.pdf'));plt.close()
    ### Save results
    print_results(stel, initial_obj)#, Print=False)
    plt.savefig(os.path.join(OUT_DIR,f'order1_opt_nfp{stel.nfp}.pdf'))
    plt.close()
    plt.figure();plt.xlabel('Function evaluation')
    plt.plot(np.array(obj_array)[len(obj_array)//2:]);plt.ylabel('Objective function')
    plt.savefig(os.path.join(OUT_DIR,f'objective_function_nfp{stel.nfp}.pdf'));plt.close()
    stel.B_contour(show=False)
    plt.savefig(os.path.join(OUT_DIR,f'Bcontour_nfp{stel.nfp}.pdf'));plt.close()
    stel.plot(show=False)
    plt.savefig(os.path.join(OUT_DIR,f'stelplot_nfp{stel.nfp}.pdf'));plt.close()
    if show:
        plt.show()
        # stel.plot_boundary()
    return stel

def assess_performance(nfp=1, r=0.1, nphi=201, delete_old=False, OUT_DIR=os.path.join(this_path,f'qic_nfp1')):
    if nfp==1: stel = optimized_configuration_nfp1(nphi)
    elif nfp==2: stel = optimized_configuration_nfp2(nphi)
    elif nfp==3: stel = optimized_configuration_nfp3(nphi)
    else: raise ValueError('Only nfp = 1, 2 and 3 allowed.')
    ## INPUT PARAMETERS
    vmec_output = os.path.join(OUT_DIR,f'wout_qic_nfp{nfp}_000_000000.nc')
    vmec_input = os.path.join(OUT_DIR,f'input.qic_nfp{nfp}')
    booz_file = os.path.join(OUT_DIR,f"boozmn_out.nc")
    neo_executable = '/Users/rogeriojorge/local/STELLOPT/NEO/Release/xneo'
    neo_in_file = os.path.join(this_path,"neo_in.out")
    ## CREATE OUTPUT DIRECTORY
    if delete_old and os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
        os.makedirs(OUT_DIR, exist_ok=True)
    os.chdir(OUT_DIR)
    ## CREATE VMEC INPUT FILE
    stel.to_vmec(vmec_input, r=r, ntorMax=50, params={'mpol':6, 'ntor': 16, 'ns_array': [51, 101], 'ftol_array':[1e-14, 1e-14], "niter_array":[4000,20000]})
    ## RUN VMEC
    from simsopt.util import MpiPartition
    mpi = MpiPartition()
    # try: vmec = Vmec(vmec_output, mpi=mpi)
    vmec = Vmec(vmec_input, mpi=mpi)
    vmec.indata.ns_array[:2]    = [51,101]
    vmec.indata.niter_array[:2] = [4000,20000]
    vmec.indata.ftol_array[:2]  = [1e-14,1e-14]
    vmec.run()
    ## PLOT VMEC
    try: vmecPlot2.main(file=vmec.output_file, name=f'qic_nfp{nfp}', figures_folder=OUT_DIR)
    except Exception as e: print(e)
    ## RUN BOOZ_XFORM
    b1 = Boozer(vmec, mpol=51, ntor=51)
    boozxform_nsurfaces=10
    booz_surfaces = np.linspace(0,1,boozxform_nsurfaces,endpoint=False)
    b1.register(booz_surfaces)
    b1.run()
    b1.bx.write_boozmn(booz_file)
    ## PLOT BOOZ_XFORM
    fig = plt.figure(); bx.surfplot(b1.bx, js=1,  fill=False, ncontours=35)
    plt.savefig(os.path.join(OUT_DIR, f"Boozxform_surfplot_1_qic_nfp{nfp}.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
    fig = plt.figure(); bx.surfplot(b1.bx, js=int(boozxform_nsurfaces/2), fill=False, ncontours=35)
    plt.savefig(os.path.join(OUT_DIR, f"Boozxform_surfplot_2_qic_nfp{nfp}.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
    fig = plt.figure(); bx.surfplot(b1.bx, js=boozxform_nsurfaces-1, fill=False, ncontours=35)
    plt.savefig(os.path.join(OUT_DIR, f"Boozxform_surfplot_3_qic_nfp{nfp}.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
    fig = plt.figure(); bx.symplot(b1.bx, helical_detail = True, sqrts=True)
    plt.savefig(os.path.join(OUT_DIR, f"Boozxform_symplot_qic_nfp{nfp}.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
    fig = plt.figure(); bx.modeplot(b1.bx, sqrts=True); plt.xlabel(r'$s=\psi/\psi_b$')
    plt.savefig(os.path.join(OUT_DIR, f"Boozxform_modeplot_qic_nfp{nfp}.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
    ## RUN NEO
    shutil.copyfile(neo_in_file,os.path.join(OUT_DIR,'neo_in.out'))
    bashCommand = f'{neo_executable} out'
    run_neo = subprocess.Popen(bashCommand.split())
    run_neo.wait()
    token = open(os.path.join(OUT_DIR,'neo_out.out'),'r')
    linestoken=token.readlines()
    eps_eff=[]
    s_radial=[]
    for x in linestoken:
        s_radial.append(float(x.split()[0])/51)
        eps_eff.append(float(x.split()[1])**(2/3))
    token.close()
    s_radial = np.array(s_radial)
    eps_eff = np.array(eps_eff)
    s_radial = s_radial[np.argwhere(~np.isnan(eps_eff))[:,0]]
    eps_eff = eps_eff[np.argwhere(~np.isnan(eps_eff))[:,0]]
    fig = plt.figure(figsize=(7, 3), dpi=200)
    ax = fig.add_subplot(111)
    plt.plot(s_radial,eps_eff)
    ax.set_yscale('log')
    plt.xlabel(r'$s=\psi/\psi_b$', fontsize=12)
    plt.ylabel(r'$\epsilon_{eff}$', fontsize=14)
    plt.tight_layout()
    fig.savefig(f'neo_out_pyqic_nfp{nfp}.pdf', dpi=fig.dpi)#, bbox_inches = 'tight', pad_inches = 0)
    ## RUN SIMPLE
    nparticles = 1500  # number of particles
    tfinal = 1e-3  # seconds
    nsamples = 10000  # number of time steps
    s_initial = 0.25 # Same s_initial as precise quasisymmetry paper
    B_scale = 5.7/vmec.wout.b0  # Scale the magnetic field by a factor
    Aminor_scale = 1.7/vmec.wout.Aminor_p  # Scale the machine size by a factor
    notrace_passing = 0  # If 1 skip tracing of passing particles
    g_field = Simple(wout_filename=vmec.output_file, B_scale=B_scale, Aminor_scale=Aminor_scale)
    g_particle = ChargedParticleEnsemble(r_initial=s_initial)
    print("Starting particle tracer")
    g_orbits = ParticleEnsembleOrbit_Simple(
        g_particle, g_field, tfinal=tfinal,
        nparticles=nparticles, nsamples=nsamples,
        notrace_passing=notrace_passing,
    )
    print(f"  Final loss fraction = {g_orbits.total_particles_lost}")
    ## PLOT SIMPLE
    g_orbits.plot_loss_fraction(show=False, save=True)
    data=np.column_stack([g_orbits.time, g_orbits.loss_fraction_array])
    datafile_path='./loss_history.dat'
    np.savetxt(datafile_path, data, fmt=['%s','%s'])
    ## RETURN TO MAIN DIRECTORY
    os.chdir(this_path)

if __name__ == "__main__":
    stel = optimized_configuration_nfp1()
    parser = argparse.ArgumentParser()
    parser.add_argument("--nfp", type=int, default=1, required=False)
    parser.add_argument("--nphi", type=int, default=stel.nphi, required=False)
    parser.add_argument("--B01", type=float, default=0.18, required=False)
    parser.add_argument("--niterations", type=int, default=300, required=False)
    parser.add_argument('--assess_performance', action='store_true')
    parser.add_argument('--no-assess_performance', dest='refine_optimized', action='store_false')
    parser.set_defaults(assess_performance=False)
    parser.add_argument("--r_plot", type=float, default=0.09, required=False)
    parser.add_argument('--refine_optimization', action='store_true')
    parser.add_argument('--no-refine_optimization', dest='refine_optimized', action='store_false')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--no-plot', dest='refine_optimized', action='store_false')
    parser.add_argument("--ndspline", type=int, default=6, required=False)
    parser.set_defaults(feature=False)
    args = parser.parse_args()
    ## CREATE OUTPUT DIRECTORY
    OUT_DIR = os.path.join(this_path,f'qic_nfp{args.nfp}')
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.plot:
        stel = optimized_configuration_nfp1() if args.nfp == 1 else optimized_configuration_nfp2() if args.nfp==2 else optimized_configuration_nfp3()
        stel.plot_boundary(args.r_plot,)
        stel.B_contour(args.r_plot)
        exit()
    if args.assess_performance:
        assess_performance(r=args.r_plot, nfp=args.nfp, delete_old=False, nphi=args.nphi, OUT_DIR=OUT_DIR)
    else:
        stel = main(nfp=args.nfp, refine_optimization=args.refine_optimization, nphi=args.nphi, maxiter = args.niterations, B01=args.B01, N_d_over_curvature_spline=args.ndspline, OUT_DIR=OUT_DIR, show=False if args.assess_performance else True)