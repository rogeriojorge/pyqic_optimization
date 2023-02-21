from qic import Qic
def optimized_configuration_nfp1(nphi=131,order = "r2"):
    rc      = [1.0,0.0,-0.2439559195801846,0.0,-0.005902401785212318,0.0,0.008651903466203575]
    zs      = [0.0,0.0,-0.2621364614557283,0.0,0.022714559645688404,0.0,0.0009428809017000957]
    B0_vals = [1.0,0.21559571183196047]
    omn_method = "non-zone-fourier"
    k_buffer = 1
    p_buffer = 2
    d_over_curvature_cvals = [0.5625200227861852,0.0017908279772609849,0.001564219083469506,0.0023546349006158714,0.003594935425409211,-0.0022870698459557044]
    delta   = 0.1
    d_svals = []
    nfp     = 1
    iota    = -0.5814380165427828
    X2s_svals = []
    X2c_cvals = [0.0]
    X2s_cvals = [-0.0008469603739260727,0.008041155094850153,-0.012057962355175965,-0.015153904312153903,0.0004552306217611255,0.0182334484157319,-0.003198700548781433]
    X2c_svals = [0.0,0.011690566820815557,0.006481207601449105,0.007110303537469069,0.0050940330348737545,0.008342428955914395,0.0007775926386645675,-0.007520077069291975]
    p2      = 0.0
    # B20QI_deviation_max = 0.0013239222344068047
    # B2cQI_deviation_max = 6.7401397392594316
    # B2sQI_deviation_max = 0.003622080216838519
    # Max |X20| = 0.38867331268173333
    # Max |Y20| = 0.6701958789604656
    # gradgradB inverse length: 5.41148983382917
    # d2_volume_d_psi2 = 319.45525430102543
    # max curvature_d(0) = 1.5452894496699363
    # max d_d(0) = 0.8800841083055937
    # max gradB inverse length: 2.559561557030413
    # Max elongation = 4.984710726275307
    # Initial objective = 157382658004.13913
    # Final objective = 2160.0967061773763
    return Qic(omn_method = omn_method, delta=delta, p_buffer=p_buffer, p2=p2, k_buffer=k_buffer, rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, B2c_cvals=X2c_cvals, B2s_svals=X2s_svals, order=order, d_over_curvature_cvals=d_over_curvature_cvals, B2c_svals=X2c_svals, B2s_cvals=X2s_cvals)