from qic import Qic
def optimized_configuration_nfp1(nphi=131,order = "r2"):
    rc      = [1.0,0.0,-0.2442340495591051,0.0,0.012246766510806244,0.0,0.0003506815435626857]
    zs      = [0.0,0.0,-0.21529867608246755,0.0,0.009866813285174771,0.0,0.0006589280293249928]
    B0_vals = [1.0,0.38459473412823864]
    omn_method = "non-zone-fourier"
    k_buffer = 1
    p_buffer = 2
    d_over_curvature_cvals = [0.8108801196203264,0.025137799029549428,0.024961996730818642]
    delta   = 0.1
    d_svals = []
    nfp     = 1
    iota    = -0.44464632925343595
    X2s_svals = []
    X2c_cvals = [0.0]
    X2s_cvals = [0.15229928234357484,0.35462352920903606,0.20989065113723473]
    X2c_svals = [0.0,0.06874027899629963,0.09111538448665989,0.3755570629030528]
    p2      = 0.0
    # B20QI_deviation_max = 3.1119201636120586e-06
    # B2cQI_deviation_max = 0.5916769625085219
    # B2sQI_deviation_max = 2.7858419157089642e-06
    # Max |X20| = 0.7132040015887711
    # Max |Y20| = 0.7014758443803538
    # gradgradB inverse length: 2.5283720862774772
    # d2_volume_d_psi2 = 307.73333941260756
    # max curvature_d(0) = 1.774494061135606
    # max d_d(0) = 1.5277159865657974
    # max gradB inverse length: 1.5738146959617099
    # Max elongation = 4.505891873293163
    # Initial objective = 114.69851967977534
    # Final objective = 101.3354775904415
    return Qic(omn_method = omn_method, delta=delta, p_buffer=p_buffer, p2=p2, k_buffer=k_buffer, rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, B2c_cvals=X2c_cvals, B2s_svals=X2s_svals, order=order, d_over_curvature_cvals=d_over_curvature_cvals, B2c_svals=X2c_svals, B2s_cvals=X2s_cvals)