from qic import Qic
def optimized_configuration_nfp1(nphi=131,order = "r2"):
    rc      = [1.0,0.0,-0.4043530784231576,0.0,0.07796989770318574,0.0,-0.0082087261848208,0.0,0.0]
    zs      = [0.0,0.0,-0.28170515645136573,0.0,0.07489395356568052,0.0,-0.008639693243704984,0.0,-0.0008771660185332342]
    B0_vals = [1.0,0.23917536708517095]
    omn_method = "non-zone-fourier"
    k_buffer = 1
    p_buffer = 2
    d_over_curvature_cvals = [0.4862371848884042,-0.00010182544360597817,0.08413289840672383,0.0020309972182432348,0.010274873318007867,-0.007138714072450926,0.0011484005727599674]
    delta   = 0.1
    d_svals = []
    nfp     = 1
    iota    = -0.6737673855885551
    X2s_svals = []
    X2c_cvals = [0.0]
    X2s_cvals = [-0.6271500414819506,-0.02256964765045181,-0.018529490303029625]
    X2c_svals = [0.12838986197355845,0.001726448479794137,-0.008327796453206457]
    p2      = 0.0
    # B20QI_deviation_max = 0.0008180426186374379
    # B2cQI_deviation_max = 0.9108071953255299
    # B2sQI_deviation_max = 0.0066976868535926215
    # Max |X20| = 0.2566216444280103
    # Max |Y20| = 4.590465644855988
    # Max |X3c1| = 0.10330990818888183
    # gradgradB inverse length: 3.085753958818333
    # d2_volume_d_psi2 = 98.13908466516114
    # max curvature_d(0) = 0.8996195319922696
    # max d_d(0) = 0.5186290568762595
    # max gradB inverse length: 1.5451570170055415
    # Max elongation = 6.9079245286286195
    # Initial objective = 162.675311372933
    # Final objective = 152.88196086017373
    return Qic(omn_method = omn_method, delta=delta, p_buffer=p_buffer, p2=p2, k_buffer=k_buffer, rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, B2c_cvals=X2c_cvals, B2s_svals=X2s_svals, order=order, d_over_curvature_cvals=d_over_curvature_cvals, B2c_svals=X2c_svals, B2s_cvals=X2s_cvals)