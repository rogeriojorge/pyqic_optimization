from qic import Qic
def optimized_configuration_nfp1(nphi=131,order = "r2"):
    rc      = [1.0,0.0,-0.2453325697412797,0.0,0.010788083520978094,0.0,0.0011693359148586733]
    zs      = [0.0,0.0,-0.2863213979463961,0.0,0.019753657045194925,0.0,0.0006124642434145551]
    B0_vals = [1.0,0.21008098403362757]
    omn_method = "non-zone-fourier"
    k_buffer = 1
    p_buffer = 2
    d_over_curvature_cvals = [0.6625711485252757,-0.09288900981467785,-0.18490194868923365,0.07398132547274275,0.07858542253885457,0.00703594077536164]
    delta   = 0.1
    d_svals = []
    nfp     = 1
    iota    = -0.5706353265952288
    X2s_svals = []
    X2c_cvals = [0.0]
    X2s_cvals = [-0.3149589745169444,-0.11559893698684323,0.12200104291601922,0.41132192320103705,0.19544234850891232,0.5031047933385429,0]
    X2c_svals = [0.0,0.73675524778851,1.8102107594756789,1.5697811510976547,0.20047152195120066,-0.6162541594404221,-0.08094193337529643,0]
    p2      = 0.0
    # B20QI_deviation_max = 3.379897305111346e-05
    # B2cQI_deviation_max = 1.2338918585397196
    # B2sQI_deviation_max = 2.5836826824610082e-05
    # Max |X20| = 3.1816776858453655
    # Max |Y20| = 2.219381032406734
    # gradgradB inverse length: 4.414067146587469
    # d2_volume_d_psi2 = 158.99248434509224
    # max curvature_d(0) = 1.923039762854857
    # max d_d(0) = 1.046619643189416
    # max gradB inverse length: 1.6749557313143786
    # Max elongation = 5.069772511817071
    # Initial objective = 13.83350156901147
    # Final objective = 13.367911322631898
    return Qic(omn_method = omn_method, delta=delta, p_buffer=p_buffer, p2=p2, k_buffer=k_buffer, rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, B2c_cvals=X2c_cvals, B2s_svals=X2s_svals, order=order, d_over_curvature_cvals=d_over_curvature_cvals, B2c_svals=X2c_svals, B2s_cvals=X2s_cvals)