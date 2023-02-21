from qic import Qic
def optimized_configuration_nfp2(nphi=131,order = "r2"):
    rc      = [1.0,0.0,-0.08918655985392808,0.0,0.009081758190992023,0.0,-0.0005113294130876145]
    zs      = [0.0,0.0,-0.08430553630578748,0.0,0.010892886553601802,0.0,-0.0003177640880228542]
    B0_vals = [1.0,0.18899158317327042]
    omn_method = "non-zone-fourier"
    k_buffer = 1
    p_buffer = 2
    d_over_curvature_cvals = [0.5654394394391244,-0.1633658435159607,-0.07173208957259546,1.7314126341981982e-05,3.061087646672029e-05,-8.110862722195534e-06]
    delta   = 0.1
    d_svals = []
    nfp     = 2
    iota    = -1.1796737490311668
    X2s_svals = []
    X2c_cvals = [0.0]
    X2s_cvals = [-0.2734043657439222,1.4065888058702816,0.029745558294259906,-1.0629167261200834e-05,-7.267528013785432e-06,2.7513284407228802e-05,1.5204503958222569e-05]
    X2c_svals = [0.0,1.4013944919714798,1.9769008977853662,-0.05468685080430061,-8.068870569021699e-06,-7.364996526771852e-06,2.2655370214237405e-05,4.440360079760122e-06]
    p2      = 0.0
    # B20QI_deviation_max = 0.0001281156289373031
    # B2cQI_deviation_max = 1.1190840223667333
    # B2sQI_deviation_max = 0.00029618136973208475
    # Max |X20| = 3.4748863300708637
    # Max |Y20| = 6.978315111113291
    # gradgradB inverse length: 5.292948303017765
    # d2_volume_d_psi2 = 340.97462322134265
    # max curvature_d(0) = 0.8573071587196209
    # max d_d(0) = 0.2834130701511009
    # max gradB inverse length: 3.072033173833159
    # Max elongation = 7.70531238562021
    # Initial objective = 1373617821535433.5
    # Final objective = 65487.480289949744
    return Qic(omn_method = omn_method, delta=delta, p_buffer=p_buffer, p2=p2, k_buffer=k_buffer, rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, B2c_cvals=X2c_cvals, B2s_svals=X2s_svals, order=order, d_over_curvature_cvals=d_over_curvature_cvals, B2c_svals=X2c_svals, B2s_cvals=X2s_cvals)