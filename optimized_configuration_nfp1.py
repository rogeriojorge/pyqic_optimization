from qic import Qic
def optimized_configuration_nfp1(nphi=131,order = "r2"):
    rc      = [1.0,0.0,-0.3769840546845129,0.0,0.06638943558536783,0.0,-0.006586490041315909]
    zs      = [0.0,0.0,-0.25174076757431,0.0,0.05197865874262495,0.0,-0.005775383353138784]
    B0_vals = [1.0,0.1829554227244039]
    omn_method = "non-zone-fourier"
    k_buffer = 1
    p_buffer = 2
    d_over_curvature = 0.719113863340804
    delta   = 0.1
    d_svals = [0.0,-0.28255935369015794,-0.008791845038436464,0.05501067203309419]
    nfp     = 1
    iota    = -0.5777906200461982
    X2s_svals = [0.0,0.0,0.0,0.0]
    X2c_cvals = [0.0,0.0,0.0]
    X2s_cvals = [-0.062070563029839154,-0.9187686562342332,0.26509305725825827]
    X2c_svals = [0.0,0.36992967608193633,0.015583633553770782,-0.0018364902447311213]
    p2      = 0.0
    # B20QI_deviation_max = 0.08200763645232655
    # B2cQI_deviation_max = 0.9696817243001541
    # B2sQI_deviation_max = 0.2908936003834679
    # Max |X20| = 0.7682980593423085
    # Max |Y20| = 2.0490995167002057
    # gradgradB inverse length: 1.8383160499175664
    # d2_volume_d_psi2 = 233.3622907515337
    # max curvature_d(0) = 1.2222517853041275
    # max d_d(0) = 0.743764023100379
    # max gradB inverse length: 1.6135438762770686
    # Max elongation = 5.515388534962833
    # Initial objective = 7.780315912497617
    # Final objective = 7.16195559419636
    return Qic(omn_method = omn_method, delta=delta, p_buffer=p_buffer, p2=p2, k_buffer=k_buffer, rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, B2c_cvals=X2c_cvals, B2s_svals=X2s_svals, order=order, d_over_curvature=d_over_curvature, B2c_svals=X2c_svals, B2s_cvals=X2s_cvals)