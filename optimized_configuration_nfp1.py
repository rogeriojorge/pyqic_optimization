from qic import Qic
def optimized_configuration_nfp1(nphi=131,order = "r2"):
    rc      = [1.0,0.0,-0.24520170423490217,0.0,0.010765145294508918,0.0,0.0011621905721043038]
    zs      = [0.0,0.0,-0.28737119132174016,0.0,0.01969872034972122,0.0,0.0006200867634944894]
    B0_vals = [1.0,0.22011717222035598]
    omn_method = "non-zone-fourier"
    k_buffer = 1
    p_buffer = 2
    d_over_curvature_cvals = [0.6739705641192966,-0.09290423581177035,-0.18856345749135037,0.07494007753393635,0.07997265828886044,0.007044896708618735]
    delta   = 0.1
    d_svals = []
    nfp     = 1
    iota    = -0.5635417466075828
    X2s_svals = []
    X2c_cvals = [0.0]
    X2s_cvals = [-0.17837139317192113,-0.25528010284552227,0.21844645995790493,0.8959616960249079,0.5300759213406172,0.5499538423368244,-0.00241143311953455]
    X2c_svals = [0.0,0.7524617876376685,1.3925594655705595,1.2993408336460845,0.4225136090003433,-0.3810615593801048,-0.004601719414758445,-0.0027384563079544838]
    p2      = 0.0
    # B20QI_deviation_max = 0.0002756776445387299
    # B2cQI_deviation_max = 1.1832813029211005
    # B2sQI_deviation_max = 0.00021118355002025524
    # Max |X20| = 2.695809505711514
    # Max |Y20| = 3.2642184873070614
    # gradgradB inverse length: 3.8152299404201413
    # d2_volume_d_psi2 = 183.31190132080303
    # max curvature_d(0) = 1.9081716500045143
    # max d_d(0) = 1.0572947844401073
    # max gradB inverse length: 1.6703774776265896
    # Max elongation = 4.857159273103788
    # Initial objective = 26.162611903336664
    # Final objective = 14.589635580588249
    return Qic(omn_method = omn_method, delta=delta, p_buffer=p_buffer, p2=p2, k_buffer=k_buffer, rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, B2c_cvals=X2c_cvals, B2s_svals=X2s_svals, order=order, d_over_curvature_cvals=d_over_curvature_cvals, B2c_svals=X2c_svals, B2s_cvals=X2s_cvals)