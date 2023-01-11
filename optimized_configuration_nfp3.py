from qic import Qic
def optimized_configuration_nfp3(nphi=131,order = "r2"):
    rc      = [1.0,0.0,-1.3619729314654734,0.0,0.5396042253308437,0.0,-0.08876804371923024]
    zs      = [0.0,0.0,-3.5988340117453133,0.0,0.6101344436630362,0.0,-0.2865372063933923]
    B0_vals = [1.0,0.25663288201347745]
    omn_method = "non-zone-fourier"
    k_buffer = 1
    p_buffer = 2
    d_over_curvature = 0.408530826716179
    delta   = 0.1
    d_svals = [0.0,0.05042518627679753,0.0006613505788215185,-0.016261229037755974]
    nfp     = 3
    iota    = -4.790436775972063
    X2s_svals = [0.0,0.0,0.0,0.0]
    X2c_cvals = [0.0,0.0,0.0]
    X2s_cvals = [0.2148408203155872,-0.08387934375600088,-0.028392369139176647]
    X2c_svals = [0.0,1.3404399322263894,-0.6127446440863573,-1.2617508434216993]
    p2      = 0.0
    # B20QI_deviation_max = 0.3826408653599289
    # B2cQI_deviation_max = 1.8129708285206825
    # B2sQI_deviation_max = 0.5375127206144379
    # Max |X20| = 2.4052106830326885
    # Max |Y20| = 2.8654732262191067
    # gradgradB inverse length: 3.553091629115655
    # d2_volume_d_psi2 = 138.91617048277797
    # max curvature_d(0) = 0.01144094968738099
    # max d_d(0) = 0.013824518190520994
    # max gradB inverse length: 1.4932796683513165
    # Max elongation = 10.695785062960393
    # Initial objective = 30.629513510601107
    # Final objective = 13.158749649651783
    return Qic(omn_method = omn_method, delta=delta, p_buffer=p_buffer, p2=p2, k_buffer=k_buffer, rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, B2c_cvals=X2c_cvals, B2s_svals=X2s_svals, order=order, d_over_curvature=d_over_curvature, B2c_svals=X2c_svals, B2s_cvals=X2s_cvals)