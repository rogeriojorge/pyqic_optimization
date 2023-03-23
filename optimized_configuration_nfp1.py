from qic import Qic
def optimized_configuration_nfp1(nphi=131,order = "r3"):
    rc      = [1.0,0.0,-0.40419493727378225,0.0,0.07749858323652713,0.0,-0.008013546720325672,0.0,0.0]
    zs      = [0.0,0.0,-0.29912426609580345,0.0,0.07739751992932789,0.0,-0.008553580758964657,0.0,-0.0010096109818069788]
    B0_vals = [1.0,0.22445287461531135]
    omn_method = "non-zone-fourier"
    k_buffer = 1
    p_buffer = 2
    d_over_curvature_cvals = []
    d_over_curvature_spline = [0.4999745964106197,0.49994693815466923,0.49994222722710435,0.5003212567818612,0.4999845717332855,0.49961047627126587,0.5022500356152756,0.5047105617598807,0.5000214999011241,0.49685638871039695,0.4995734302419541,0.503055524678494,0.5007859628753698,0.5011465711038654,0.5011639702520905,0.5025380210509655,0.5026147266650647,0.5017909740285735,0.502278985078561,0.5030606904698576,0.5009798951081692,0.5016391658400075,0.500625125689159,0.5009692502425482,0.4994770333345152,0.4996227015645849,0.500198636267599,0.5000091767358351,0.4999508722434235,0.5000205692139574]
    delta   = 0.1
    d_svals = []
    nfp     = 1
    iota    = -0.6880173944061565
    X2s_svals = []
    X2c_cvals = [0.0]
    X2s_cvals = [-0.005137965316636749,0.0010008283099320822,0.0010017707659376604]
    X2c_svals = [-0.005007353914768078,0.0010023313267335258,0.0010020574163446374]
    p2      = 0.0
    # B20QI_deviation_max = 0.08618788888872775
    # B2cQI_deviation_max = 2.762291991508029
    # B2sQI_deviation_max = 0.15587950423616992
    # Max |X20| = 0.26404242718465964
    # Max |Y20| = 0.5103594081874541
    # Max |X3c1| = 0.02611744756488051
    # gradgradB inverse length: 3.7971836502682716
    # d2_volume_d_psi2 = 169.88766007760972
    # max curvature_d(0) = 1.0152147360333639
    # max d_d(0) = 0.5075616621359004
    # max gradB inverse length: 1.5036651930606766
    # Max elongation = 5.292842548383343
    # Initial objective = 46656.167738518896
    # Final objective = 591.23150063875
    return Qic(omn_method = omn_method, delta=delta, p_buffer=p_buffer, p2=p2, k_buffer=k_buffer, rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, B2c_cvals=X2c_cvals, B2s_svals=X2s_svals, order=order, d_over_curvature_cvals=d_over_curvature_cvals, B2c_svals=X2c_svals, B2s_cvals=X2s_cvals, d_over_curvature_spline=d_over_curvature_spline)