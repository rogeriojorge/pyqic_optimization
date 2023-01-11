from qic import Qic
def optimized_configuration_nfp3(nphi=131,order = "r2"):
    rc      = [1.0,0.0,-0.9030531435616707,0.0,0.3118157631062801,0.0,-0.0393855979650117]
    zs      = [0.0,0.0,-2.28617047185919,0.0,0.35133011189776986,0.0,-0.15283373248019771]
    B0_vals = [1.0,0.431338515815339]
    omn_method = "non-zone-fourier"
    k_buffer = 1
    p_buffer = 2
    d_over_curvature = 0.4632854514121336
    delta   = 0.1
    d_svals = [0.0,0.03900204849238709,-0.002622300041810445,-0.007227863521227267]
    nfp     = 3
    iota    = -4.566183921197732
    X2s_svals = [0.0,0.0,0.0,0.0]
    X2c_cvals = [0.0,0.0,0.0]
    X2s_cvals = [-0.9844001215612406,-0.10543673670982745,-0.1657295914649798]
    X2c_svals = [0.0,1.125240670364168,-2.6720884644019227,-1.3139496021157204]
    p2      = 0.0
    # B20QI_deviation_max = 0.21851951022771887
    # B2cQI_deviation_max = 3.0765989061338295
    # B2sQI_deviation_max = 0.32243233603296123
    # Max |X20| = 2.1663340901539696
    # Max |Y20| = 7.930991044404474
    # gradgradB inverse length: 4.698330947573702
    # d2_volume_d_psi2 = 899.1981845623792
    # max curvature_d(0) = 0.08902848043841931
    # max d_d(0) = 0.07764251687512314
    # max gradB inverse length: 1.9728657919733337
    # Max elongation = 12.621228316682693
    # Initial objective = 364.5426019431311
    # Final objective = 29.973126306149915
    return Qic(omn_method = omn_method, delta=delta, p_buffer=p_buffer, p2=p2, k_buffer=k_buffer, rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, B2c_cvals=X2c_cvals, B2s_svals=X2s_svals, order=order, d_over_curvature=d_over_curvature, B2c_svals=X2c_svals, B2s_cvals=X2s_cvals)