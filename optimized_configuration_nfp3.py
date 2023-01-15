from qic import Qic
def optimized_configuration_nfp3(nphi=131,order = "r2"):
    rc      = [1.0,0.0,-0.027854122568958858,0.0,0.000259047896456429,0.0,-2.1413569029860537e-05]
    zs      = [0.0,0.0,-0.02814303428304686,0.0,0.00023367898501901738,0.0,-2.0778982053344098e-05]
    B0_vals = [1.0,0.09220527328779891]
    omn_method = "non-zone-fourier"
    k_buffer = 1
    p_buffer = 2
    d_over_curvature = 0.9221820451136327
    delta   = 0.1
    d_svals = [0.0,-0.1518302595638622,-0.006029540126025487,-0.0007772276846135682]
    nfp     = 3
    iota    = -0.7781298888460842
    X2s_svals = [0.0,0.0,0.0,0.0]
    X2c_cvals = [0.0,0.0,0.0]
    X2s_cvals = [0.031630128911416235,-0.26842801952046874,-0.707024791458418]
    X2c_svals = [0.0,-0.06069624735772538,0.6375379246337994,0.046459053318966506]
    p2      = 0.0
    # B20QI_deviation_max = 0.09426868811716327
    # B2cQI_deviation_max = 0.036224133107541245
    # B2sQI_deviation_max = 0.18090003945311953
    # Max |X20| = 0.7220255962125043
    # Max |Y20| = 2.5516410577349227
    # gradgradB inverse length: 5.011279355443843
    # d2_volume_d_psi2 = 623.0756655906639
    # max curvature_d(0) = 5.81634766611165
    # max d_d(0) = 4.865109313217104
    # max gradB inverse length: 2.9164033435051455
    # Max elongation = 2.878708446589869
    # Initial objective = 36.90065509953848
    # Final objective = 35.93371829494155
    return Qic(omn_method = omn_method, delta=delta, p_buffer=p_buffer, p2=p2, k_buffer=k_buffer, rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, B2c_cvals=X2c_cvals, B2s_svals=X2s_svals, order=order, d_over_curvature=d_over_curvature, B2c_svals=X2c_svals, B2s_cvals=X2s_cvals)