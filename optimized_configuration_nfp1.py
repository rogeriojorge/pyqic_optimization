from qic import Qic
def optimized_configuration_nfp1(nphi=131,order = "r2"):
    rc      = [1.0,0.0,-0.37762830312022766,0.0,0.06656000778828239,0.0,-0.0065778004540449275]
    zs      = [0.0,0.0,-0.2446187024317527,0.0,0.049436953635961345,0.0,-0.005260115776659658]
    B0_vals = [1.0,0.21751520043441647]
    omn_method = "non-zone-fourier"
    k_buffer = 1
    p_buffer = 2
    d_over_curvature = 0.7525257797925362
    delta   = 0.1
    d_svals = [0.0,-0.3213153841715881,-0.004092125408892158,0.05034102285517156]
    nfp     = 1
    iota    = -0.5787594633710342
    X2s_svals = [0.0,0.0,0.0,0.0]
    X2c_cvals = [0.0,0.0,0.0]
    X2s_cvals = [-0.07780028001471415,-0.13616748796986322,0.9397181048830656]
    X2c_svals = [0.0,-0.005671398328420943,0.6144318467189016,0.5603568436128983]
    p2      = 0.0
    # B20QI_deviation_max = 0.0719000841538554
    # B2cQI_deviation_max = 0.5453106916556201
    # B2sQI_deviation_max = 0.22462687497982248
    # Max |X20| = 1.677921083528997
    # Max |Y20| = 1.9288685321508305
    # gradgradB inverse length: 2.8887537170891955
    # d2_volume_d_psi2 = 291.66226919298094
    # max curvature_d(0) = 1.157682195394777
    # max d_d(0) = 0.6926752705904604
    # max gradB inverse length: 1.639725738946693
    # Max elongation = 5.272907438648189
    # Initial objective = 31.82586693564446
    # Final objective = 28.83754036307843
    return Qic(omn_method = omn_method, delta=delta, p_buffer=p_buffer, p2=p2, k_buffer=k_buffer, rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, B2c_cvals=X2c_cvals, B2s_svals=X2s_svals, order=order, d_over_curvature=d_over_curvature, B2c_svals=X2c_svals, B2s_cvals=X2s_cvals)