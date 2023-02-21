from qic import Qic
def optimized_configuration_nfp2(nphi=131,order = "r2"):
    rc      = [1.0,0.0,-0.08912900941552898,0.0,0.009104118293488075,0.0,-0.0005281002000878083]
    zs      = [0.0,0.0,-0.0835250107955374,0.0,0.010792506587320873,0.0,-0.0007555450010884172]
    B0_vals = [1.0,0.14423228770593907]
    omn_method = "non-zone-fourier"
    k_buffer = 1
    p_buffer = 2
    d_over_curvature_cvals = [0.5601281047963904,-0.17089593070531148,-0.06960164901826307,1.5100267557868352e-05,0.0005575058786309018,5.67932166154396e-05]
    delta   = 0.1
    d_svals = []
    nfp     = 2
    iota    = -1.1822986434276284
    X2s_svals = []
    X2c_cvals = [0.0]
    X2s_cvals = [-1.314106103266875,1.9277521162147901,0.28447880245376034,-0.0006661169570642102,0.0012450522564134321,-0.0032669494513762666,-0.0003192776947304741]
    X2c_svals = [0.0,0.7333788942830073,3.3971582384192023,0.3872957686374267,-0.0010976351802139273,6.609731465525193e-05,0.00014205002433269562,0.00014186280860763718]
    p2      = 0.0
    # B20QI_deviation_max = 4.338391858604851e-05
    # B2cQI_deviation_max = 1.8604852279348607
    # B2sQI_deviation_max = 3.725777901220084e-05
    # Max |X20| = 4.967255299886579
    # Max |Y20| = 14.771859117135342
    # gradgradB inverse length: 7.046000891689751
    # d2_volume_d_psi2 = 348.4896313123392
    # max curvature_d(0) = 1.664867845334639
    # max d_d(0) = 0.5334189744386632
    # max gradB inverse length: 3.1351415173405255
    # Max elongation = 8.520804634317804
    # Initial objective = 3192.6039031268947
    # Final objective = 616.067812948329
    return Qic(omn_method = omn_method, delta=delta, p_buffer=p_buffer, p2=p2, k_buffer=k_buffer, rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, B2c_cvals=X2c_cvals, B2s_svals=X2s_svals, order=order, d_over_curvature_cvals=d_over_curvature_cvals, B2c_svals=X2c_svals, B2s_cvals=X2s_cvals)