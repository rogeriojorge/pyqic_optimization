from qic import Qic
def optimized_configuration_nfp1(nphi=131,order = "r2"):
    rc      = [1.0,0.0,-0.3753443144761014,0.0,0.06405129100340776,0.0,-0.0057337939102006736]
    zs      = [0.0,0.0,-0.2469920960060379,0.0,0.05040443359139635,0.0,-0.004418075712882462]
    B0_vals = [1.0,0.18737684417358508]
    omn_method = "non-zone-fourier"
    k_buffer = 1
    p_buffer = 2
    d_over_curvature_cvals = [0.6235900419142353,-0.020692694945057398,-0.01814597890293869]
    delta   = 0.1
    d_svals = []
    nfp     = 1
    iota    = -0.5545305454174287
    X2s_svals = []
    X2c_cvals = [0.0]
    X2s_cvals = [-0.028196647044676622,-0.08009772375594307,-0.18106340096063728]
    X2c_svals = [0.0,1.193055838239637,1.764549648408996,0.4223425569681274]
    p2      = 0.0
    # B20QI_deviation_max = 0.0008662817596027383
    # B2cQI_deviation_max = 1.1527912908595859
    # B2sQI_deviation_max = 0.003323031672838539
    # Max |X20| = 1.381383701641869
    # Max |Y20| = 2.5371412931902992
    # gradgradB inverse length: 3.227202016797077
    # d2_volume_d_psi2 = 122.16096196790494
    # max curvature_d(0) = 0.7189442257253607
    # max d_d(0) = 0.42042872576367474
    # max gradB inverse length: 1.6221387438139099
    # Max elongation = 4.076329100126041
    # Initial objective = 4.852475056109592
    # Final objective = 4.678917065675533
    return Qic(omn_method = omn_method, delta=delta, p_buffer=p_buffer, p2=p2, k_buffer=k_buffer, rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, B2c_cvals=X2c_cvals, B2s_svals=X2s_svals, order=order, d_over_curvature_cvals=d_over_curvature_cvals, B2c_svals=X2c_svals, B2s_cvals=X2s_cvals)