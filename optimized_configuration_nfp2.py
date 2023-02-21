from qic import Qic
def optimized_configuration_nfp2(nphi=131,order = "r2"):
    rc      = [1.0,0.0,-0.08917273509254975,0.0,0.009094282095360146,0.0,-0.0005185644112073359]
    zs      = [0.0,0.0,-0.08265803252211759,0.0,0.010950093467842995,0.0,-0.0006896420805898686]
    B0_vals = [1.0,0.13984791735269353]
    omn_method = "non-zone-fourier"
    k_buffer = 1
    p_buffer = 2
    d_over_curvature_cvals = [0.5587096429831517,-0.1668646197571534,-0.06913029681668217,6.728696288100747e-06,0.00015849088119814961,4.640625605762949e-05]
    delta   = 0.1
    d_svals = []
    nfp     = 2
    iota    = -1.1832562684066559
    X2s_svals = []
    X2c_cvals = [0.0]
    X2s_cvals = [0.39027837207307914,-0.9006229129027883,-0.10478290677707153,5.439370606277695e-05,0.00015869128611556432,-0.00036389230154016934,-0.00013707231136752727]
    X2c_svals = [0.0,6.3176189568571175,-10.301720090108645,-3.4542016922224565,-0.00010173256974977657,-1.1388536694942415e-05,1.7256702295708735e-05,3.919303014741236e-05]
    p2      = 0.0
    # B20QI_deviation_max = 2.25374595270722e-05
    # B2cQI_deviation_max = 18.606816966463445
    # B2sQI_deviation_max = 3.9231344658219314e-05
    # Max |X20| = 11.603619493377664
    # Max |Y20| = 18.33645902400628
    # gradgradB inverse length: 10.6428053040541
    # d2_volume_d_psi2 = -868.2014447716143
    # max curvature_d(0) = 1.415046939605936
    # max d_d(0) = 0.4571483096535647
    # max gradB inverse length: 3.1296152049199173
    # Max elongation = 8.412907546142614
    # Initial objective = 65497.45994472972
    # Final objective = 3192.588246345628
    return Qic(omn_method = omn_method, delta=delta, p_buffer=p_buffer, p2=p2, k_buffer=k_buffer, rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, B2c_cvals=X2c_cvals, B2s_svals=X2s_svals, order=order, d_over_curvature_cvals=d_over_curvature_cvals, B2c_svals=X2c_svals, B2s_cvals=X2s_cvals)