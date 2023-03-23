from qic import Qic
def optimized_configuration_nfp2(nphi=131,order = "r2"):
    rc      = [1.0,0.0,-0.3568969204153243,0.0,0.06620803194955616,0.0,0.005267072898892156,0.0,0.0]
    zs      = [0.0,0.0,-0.5528399377206339,0.0,0.08896077079803161,0.0,-0.0017918650011510603,0.0,-0.0009761045895102616]
    B0_vals = [1.0,0.37063979826699145]
    omn_method = "non-zone-fourier"
    k_buffer = 1
    p_buffer = 2
    d_over_curvature_cvals = [0.39969561779242246,-0.0338701455928614,0.13272779496348347,-0.04980158194757328,0.027034848274715605,-0.02511811158068389,0.0011519388403890465]
    delta   = 0.1
    d_svals = []
    nfp     = 2
    iota    = -1.2664878330152007
    X2s_svals = []
    X2c_cvals = [0.0]
    X2s_cvals = [0.0348573726581137,-0.0004885538351722881,0.004285492261936712]
    X2c_svals = [-0.1657678613134016,0.0025879041622606917,-0.0014893570785638028]
    p2      = 0.0
    # B20QI_deviation_max = 0.12799729545225416
    # B2cQI_deviation_max = 6.353651028462765
    # B2sQI_deviation_max = 1.3992355065122253
    # Max |X20| = 0.24398429060128
    # Max |Y20| = 2.5523423997214807
    # Max |X3c1| = 0.2397185016152281
    # gradgradB inverse length: 12.154822528576908
    # d2_volume_d_psi2 = 1361.8272775444668
    # max curvature_d(0) = 0.6686492788179291
    # max d_d(0) = 0.3013718972197963
    # max gradB inverse length: 3.3067215922722686
    # Max elongation = 18.14461585336697
    # Final objective = 2361.532606769884
    return Qic(omn_method = omn_method, delta=delta, p_buffer=p_buffer, p2=p2, k_buffer=k_buffer, rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, B2c_cvals=X2c_cvals, B2s_svals=X2s_svals, order=order, d_over_curvature_cvals=d_over_curvature_cvals, B2c_svals=X2c_svals, B2s_cvals=X2s_cvals)