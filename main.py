#!/usr/bin/env python3
import os
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
from simsopt.mhd import Vmec, Boozer
from simsopt.util import MpiPartition
from neat.fields import Simple
from neat.tracing import ChargedParticleEnsemble, ParticleEnsembleOrbit_Simple
try: from optimized_configuration_nfp1 import optimized_configuration_nfp1
except: pass
try: from optimized_configuration_nfp2 import optimized_configuration_nfp2
except: pass
try: from optimized_configuration_nfp3 import optimized_configuration_nfp3
except: pass
mpi = MpiPartition()
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

def initial_configuration(nphi=131,order = 'r2',nfp=1):
    rc      = [ 1.0,0.0,-0.3,0.0,0.01,0.0,0.001 ]
    zs      = [ 0.0,0.0,-0.2,0.0,0.01,0.0,0.001 ]
    B0_vals = [ 1.0,0.16 ]
    omn_method ='non-zone-fourier'
    k_buffer = 1
    p_buffer = 2
    delta = 0.1
    d_over_curvature_cvals = [0.5,0.001,0.001]
    X2s_cvals = [ 0.01,0.01,0.01 ]
    X2c_svals = [ 0.0,0.01,0.01,0.01 ]
    return Qic(omn_method = omn_method, delta=delta, p_buffer=p_buffer, k_buffer=k_buffer, rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, nphi=nphi, omn=True, order=order, d_over_curvature_cvals=d_over_curvature_cvals, B2c_svals=X2c_svals, B2s_cvals=X2s_cvals)

def print_results(stel,initial_obj=0):
    out_txt  = f'from qic import Qic\n'
    out_txt += f'def optimized_configuration_nfp{stel.nfp}(nphi=131,order = "r2"):\n'
    out_txt += f'    rc      = [{",".join([str(elem) for elem in stel.rc])}]\n'
    out_txt += f'    zs      = [{",".join([str(elem) for elem in stel.zs])}]\n'
    out_txt += f'    B0_vals = [{",".join([str(elem) for elem in stel.B0_vals])}]\n'
    out_txt += f'    omn_method = "{stel.omn_method}"\n'
    out_txt += f'    k_buffer = {stel.k_buffer}\n'
    out_txt += f'    p_buffer = {stel.p_buffer}\n'
    out_txt += f'    d_over_curvature_cvals = [{",".join([str(elem) for elem in stel.d_over_curvature_cvals])}]\n'
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
    out_txt += f'    return Qic(omn_method = omn_method, delta=delta, p_buffer=p_buffer, p2=p2, k_buffer=k_buffer, rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, B2c_cvals=X2c_cvals, B2s_svals=X2s_svals, order=order, d_over_curvature_cvals=d_over_curvature_cvals, B2c_svals=X2c_svals, B2s_cvals=X2s_cvals)'
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
        print(f"fun#{info['Nfeval']} -",
            # f"\N{GREEK CAPITAL LETTER DELTA}B20 = {stel.B20QI_deviation_max:.2f},",
            # f"\N{GREEK CAPITAL LETTER DELTA}B2s = {stel.B2sQI_deviation_max:.2f},",
            f"\N{GREEK CAPITAL LETTER DELTA}B2c = {stel.B2cQI_deviation_max:.2f},",
            f"1/L\N{GREEK CAPITAL LETTER DELTA}B = {np.max(stel.inv_L_grad_B):.2f},",
            f"1/L\N{GREEK CAPITAL LETTER DELTA}\N{GREEK CAPITAL LETTER DELTA}B = {stel.grad_grad_B_inverse_scale_length:.2f},",
            f"B0(1) = {stel.B0_vals[1]:.2f},",
            f"max(elong) = {stel.max_elongation:.2f},",
            f"max(Y20) = {max(abs(stel.Y20)):.2f},",
            f"\N{GREEK CAPITAL LETTER DELTA}\N{GREEK SMALL LETTER ALPHA} = {max(abs(stel.alpha - stel.alpha_no_buffer)):.2f},",
            f"J = {objective_function:.2f}")
        obj_array.append(objective_function)
    except Exception as e:
        print(e)
        objective_function = 1e3
    return objective_function

def obj(stel):
    weight_XYZ2 = 0.007
    weight_B0vals = 2000
    B0_well_depth = 0.18
    weight_B2c_dev = 10
    weight_d_at_0 = 1
    weight_B20cs = 0.2
    weight_gradB_scale_length = 0.04
    weight_elongation = 0.05
    weight_d = 0.5
    weight_alpha_diff = 1.0
    return weight_B2c_dev*np.sum(stel.B2cQI_deviation**2)/stel.nphi \
         + weight_gradB_scale_length*np.sum((stel.inv_L_grad_B**2 + stel.grad_grad_B_inverse_scale_length_vs_varphi**2))/stel.nphi \
         + weight_B0vals*(stel.B0_vals[1]-B0_well_depth)**2 \
         + weight_elongation*np.sum(stel.elongation**2)/stel.nphi \
         + weight_XYZ2*np.sum(stel.X20**2 + stel.X2c**2 + stel.X2s**2
                            + stel.Y20**2 + stel.Y2c**2 + stel.Y2s**2
                            + stel.Z20**2 + stel.Z2c**2 + stel.Z2s**2)/stel.nphi \
         + weight_alpha_diff*np.sum((stel.alpha - stel.alpha_no_buffer)**2)/stel.nphi \
         + weight_d*np.sum(stel.d**2)/stel.nphi \
         + weight_d_at_0*stel.d_curvature_d_varphi_at_0**2 \
         + weight_d_at_0*stel.d_d_d_varphi_at_0**2 \
         + weight_B20cs*np.sum(stel.B20**2 + stel.B2c**2 + stel.B2s**2)/stel.nphi

def main(nfp=1, refine_optimization=False, nphi=91, maxiter = 3000, show=True):
    if nfp not in [1,2,3]: raise ValueError('nfp should be 1, 2 or 3.')
    if refine_optimization:
        if nfp==1: stel = optimized_configuration_nfp1(nphi)
        elif nfp==2: stel = optimized_configuration_nfp2(nphi)
        elif nfp==3: stel = optimized_configuration_nfp3(nphi)
    else: stel = initial_configuration(nfp=nfp)
    initial_obj = obj(stel)
    initial_dofs = stel.get_dofs()
    parameters_to_change = (['zs(2)','B0(1)','B2cs(1)','B2sc(0)',
                             'zs(4)','rc(2)','B2cs(2)','B2sc(1)',
                             'zs(6)','rc(4)','B2cs(3)','B2sc(2)',
                             'd_over_curvaturec(0)','d_over_curvaturec(1)','d_over_curvaturec(2)'])
    dofs = [initial_dofs[stel.names.index(parameter)] for parameter in parameters_to_change]
    # if show: plot_results(stel)
    obj_array = []
    method = 'Nelder-Mead'
    maxfev  = maxiter
    res = minimize(fun, dofs, args=(stel, parameters_to_change, {'Nfeval':0}, obj_array), method=method, tol=1e-3, options={'maxiter': maxiter, 'maxfev': maxfev, 'disp': True})
    print_results(stel, initial_obj)
    if show:
        # plot_results(stel)
        plt.figure();plt.xlabel('Function evaluation')
        plt.plot(np.array(obj_array));plt.ylabel('Objective function')
        stel.B_contour(show=False)
        stel.plot(show=False)
        plt.show()
        # stel.plot_boundary()
    return stel

def assess_performance(nfp=1, r=0.1, nphi=201, delete_old=False):
    if nfp==1: stel = optimized_configuration_nfp1(nphi)
    elif nfp==2: stel = optimized_configuration_nfp2(nphi)
    elif nfp==3: stel = optimized_configuration_nfp3(nphi)
    else: raise ValueError('Only nfp = 1, 2 and 3 allowed.')
    ## INPUT PARAMETERS
    OUT_DIR = os.path.join(this_path,f'qic_nfp{nfp}')
    vmec_output = os.path.join(OUT_DIR,f'wout_qic_nfp{nfp}_000_000000.nc')
    vmec_input = os.path.join(OUT_DIR,f'input.qic_nfp{nfp}')
    booz_file = os.path.join(OUT_DIR,f"boozmn_out.nc")
    neo_executable = '/Users/rogeriojorge/local/STELLOPT/NEO/Release/xneo'
    neo_in_file = os.path.join(this_path,"neo_in.out")
    ## CREATE OUTPUT DIRECTORY
    if delete_old and os.path.isdir(OUT_DIR): shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.chdir(OUT_DIR)
    ## CREATE VMEC INPUT FILE
    stel.to_vmec(vmec_input, r=r)
    ## RUN VMEC
    try: vmec = Vmec(vmec_output, mpi=mpi)
    except: vmec = Vmec(vmec_input, mpi=mpi)
    vmec.indata.ns_array[:3]    = [  16,    51,    101]
    vmec.indata.niter_array[:3] = [ 4000, 10000, 20000]
    vmec.indata.ftol_array[:3]  = [1e-12, 1e-13, 1e-14]
    vmec.run()
    ## PLOT VMEC
    try: vmecPlot2.main(file=vmec.output_file, name=f'qic_nfp{nfp}', figures_folder=OUT_DIR)
    except Exception as e: print(e)
    ## RUN BOOZ_XFORM
    b1 = Boozer(vmec, mpol=64, ntor=64)
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
        s_radial.append(float(x.split()[0])/101)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--nfp", type=int, default=1, required=False)
    parser.add_argument("--nphi", type=int, default=91, required=False)
    parser.add_argument("--niterations", type=int, default=300, required=False)
    parser.add_argument('--assess_performance', action='store_true')
    parser.add_argument('--no-assess_performance', dest='refine_optimized', action='store_false')
    parser.set_defaults(assess_performance=False)
    parser.add_argument("--r_plot", type=float, default=0.1, required=False)
    parser.add_argument('--refine_optimization', action='store_true')
    parser.add_argument('--no-refine_optimization', dest='refine_optimized', action='store_false')
    parser.set_defaults(feature=False)
    args = parser.parse_args()
    stel = main(nfp=args.nfp, refine_optimization=args.refine_optimization, nphi=args.nphi, maxiter = args.niterations, show=False if args.assess_performance else True)
    if args.assess_performance: assess_performance(r=args.r_plot, nfp=args.nfp, delete_old=True)