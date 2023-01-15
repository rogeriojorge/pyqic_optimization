from qic import Qic
def optimized_configuration_nfp2(nphi=131,order = "r2"):
    rc      = [1.0,0.0,-0.07268022957532712,0.0,0.004158373815471744,0.0,-0.00023951996706967138]
    zs      = [0.0,0.0,-0.06730956173070415,0.0,0.00414731590482878,0.0,-0.0002380434079538091]
    B0_vals = [1.0,0.10158099921528108]
    omn_method = "non-zone-fourier"
    k_buffer = 1
    p_buffer = 2
    d_over_curvature = 0.7834701934353268
    delta   = 0.1
    d_svals = [0.0,-0.33041767005626477,0.07999446042657082,0.07231046832658025]
    nfp     = 2
    iota    = -0.9823270275460888
    X2s_svals = [0.0,0.0,0.0,0.0]
    X2c_cvals = [0.0,0.0,0.0]
    X2s_cvals = [0.17520288862835282,-0.24915620792339987,-0.3583680564296947]
    X2c_svals = [0.0,2.1311250064933325,-0.27634587945920547,-1.0455456335215347]
    p2      = 0.0
    # B20QI_deviation_max = 0.059365489729492626
    # B2cQI_deviation_max = 0.08214716739374817
    # B2sQI_deviation_max = 0.07869514246500685
    # Max |X20| = 2.453037510213784
    # Max |Y20| = 1.3428217045673592
    # gradgradB inverse length: 3.6307218167868527
    # d2_volume_d_psi2 = 271.04133693042877
    # max curvature_d(0) = 3.143019712829905
    # max d_d(0) = 2.5552096823159394
    # max gradB inverse length: 2.67900084200771
    # Max elongation = 4.872547522173008
    # Initial objective = 24.99201558660309
    # Final objective = 22.555155290484915
    return Qic(omn_method = omn_method, delta=delta, p_buffer=p_buffer, p2=p2, k_buffer=k_buffer, rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, B2c_cvals=X2c_cvals, B2s_svals=X2s_svals, order=order, d_over_curvature=d_over_curvature, B2c_svals=X2c_svals, B2s_cvals=X2s_cvals)