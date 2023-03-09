from qic import Qic
def optimized_configuration_nfp1(nphi=131,order = "r2"):
    rc      = [1.0,0.0,-0.4043733683013211,0.0,0.07791207063670558,0.0,-0.0081794151166862,0.0,0.0]
    zs      = [0.0,0.0,-0.2815662227490585,0.0,0.07507269969913594,0.0,-0.008641187564624075,0.0,-0.0008683079527375737]
    B0_vals = [1.0,0.23184650605556334]
    omn_method = "non-zone-fourier"
    k_buffer = 1
    p_buffer = 2
    d_over_curvature_cvals = [0.5005357059995865,-0.0001310311849880901,0.0797324891117683,0.0019499272093110973,0.009978618760780256,-0.0071764023999408046,0.0012630362449780452]
    delta   = 0.1
    d_svals = []
    nfp     = 1
    iota    = -0.6607545397949548
    X2s_svals = []
    X2c_cvals = [0.0]
    X2s_cvals = [-0.5054465707787683,-0.02256964765045181,-0.018529490303029625]
    X2c_svals = [0.13310537711916648,0.001726448479794137,-0.008327796453206457]
    p2      = 0.0
    # B20QI_deviation_max = 0.0010202894859389633
    # B2cQI_deviation_max = 1.1079859022817562
    # B2sQI_deviation_max = 0.007444764305277407
    # Max |X20| = 0.1775943865024401
    # Max |Y20| = 3.3058450761170355
    # Max |X3c1| = 0.0657265042977988
    # gradgradB inverse length: 2.9031597599230485
    # d2_volume_d_psi2 = 101.62852074558295
    # max curvature_d(0) = 0.8681050552987348
    # max d_d(0) = 0.5087725505739247
    # max gradB inverse length: 1.5436765381057616
    # Max elongation = 6.465999133295312
    # Initial objective = 92.81890590036674
    # Final objective = 88.76256879665436
    return Qic(omn_method = omn_method, delta=delta, p_buffer=p_buffer, p2=p2, k_buffer=k_buffer, rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, B2c_cvals=X2c_cvals, B2s_svals=X2s_svals, order=order, d_over_curvature_cvals=d_over_curvature_cvals, B2c_svals=X2c_svals, B2s_cvals=X2s_cvals)