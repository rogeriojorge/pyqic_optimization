from qic import Qic
def optimized_configuration_nfp2(nphi=131,order = "r2"):
    rc      = [1.0,0.0,-0.0704299036947035,0.0,0.003064091902092874,0.0,-1.2811109145360494e-05]
    zs      = [0.0,0.0,-0.0672405166011886,0.0,0.003001014047860408,0.0,-2.087753869937767e-06]
    B0_vals = [1.0,0.13406736449536785]
    omn_method = "non-zone-fourier"
    k_buffer = 1
    p_buffer = 2
    d_over_curvature = 1.0019630231805818
    delta   = 0.1
    d_svals = [0.0,-0.7190552949511066,-0.05148214842848478,0.0002054102535472032]
    nfp     = 2
    iota    = -1.0498082806054503
    X2s_svals = [0.0,0.0,0.0,0.0]
    X2c_cvals = [0.0,0.0,0.0]
    X2s_cvals = [0.48701564069383907,-1.7443450588768235,-0.21242082617924785]
    X2c_svals = [0.0,2.6208135877170093,0.25756791675884017,-0.1437203647715763]
    p2      = 0.0
    # B20QI_deviation_max = 0.14204042657343408
    # B2cQI_deviation_max = 0.48141589414325214
    # B2sQI_deviation_max = 0.09986073594014698
    # Max |X20| = 0.7489989719212793
    # Max |Y20| = 14.853467782455143
    # gradgradB inverse length: 3.2506177327153774
    # d2_volume_d_psi2 = 566.8485369094154
    # max curvature_d(0) = 3.1885919498597506
    # max d_d(0) = 1.5522894771913438
    # max gradB inverse length: 2.4296381799580042
    # Max elongation = 5.258297786821333
    # Initial objective = 14.266502941361555
    # Final objective = 14.16011720867668
    return Qic(omn_method = omn_method, delta=delta, p_buffer=p_buffer, p2=p2, k_buffer=k_buffer, rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, B2c_cvals=X2c_cvals, B2s_svals=X2s_svals, order=order, d_over_curvature=d_over_curvature, B2c_svals=X2c_svals, B2s_cvals=X2s_cvals)