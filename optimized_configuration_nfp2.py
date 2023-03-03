from qic import Qic
def optimized_configuration_nfp2(nphi=131,order = "r2"):
    rc      = [1.0,0.0,-0.1162437012367244,0.0,0.017197390258696973,0.0,-0.0009771547985585418]
    zs      = [0.0,0.0,-0.2009980209669182,0.0,0.009926872969304584,0.0,0.0009677260740524507]
    B0_vals = [1.0,0.20996576553532892]
    omn_method = "non-zone-fourier"
    k_buffer = 1
    p_buffer = 2
    d_over_curvature_cvals = [0.5203875188890446,0.009835383486506314,0.0046950591442328216,-0.013029566079987676,-0.001562166509032942,-0.001656440756911019]
    delta   = 0.1
    d_svals = []
    nfp     = 2
    iota    = -1.4318953658411298
    X2s_svals = []
    X2c_cvals = [0.0]
    X2s_cvals = [0.0002944382942709811,-0.06657617495836651,-0.26832076096509216,-1.6549437854462896,-2.3724096275588513,-2.449651235616548,-2.328407514875831,-1.2490799934426358,0.5387959418159658,1.9616741883428659,2.084943465466928,1.005080530314437,-0.5501378686346159,-1.6546906028614705,-1.801936567091904,-1.1582812068178927,-0.6509308128720243,-0.36345095550227513,-0.37396522629302487,-0.3486937775329114,-0.8308440820710569,-2.014866884692511,-3.0355802821363778,-3.5261167997401253,-3.7233686358821556,-3.2835288903771422,-2.0955377004177365,-0.5811563847814014,0.7410304813609785,1.3402867782488777,2.4109473752217303,2.4576447633608294,1.5131559836995967,0.5851103013786105,0.554010025959006,0.009476213665699194,0.18835564905945157,0.19948581473572422,0.9828329973352836,2.2035078015401615,2.6354346969347384,1.6295262448524097,0.9755165559434029,-0.03745366779598659,-1.6132101057148893,-2.946970890793886,-3.6230717724947903,-3.614400688762543,-3.1926043709858956,-2.4512173405655284,-1.206685291731618,-0.5458592239991709,-0.38698311757347015,-0.3255084410644201,-0.6461943219722872,-0.8830360441354008,-1.588424871310449,-1.8594077147756858,-0.9535501162233035,0.49443632019896117,1.8631753310066679,2.1683083715865887,1.1015843799545668,-0.678919062849852,-2.0841056154563553,-2.448443266395346,-2.5100512352413977,-2.0679074801743207,-0.5684086886128618,-0.005374886040518844,0.0004901741388220653]
    X2c_svals = [-0.16740225634887074,0.22091582017527628,0.9053522886407508,1.7917904042474455,2.829911078751925,3.4365712822497416,2.8303945512014117,1.2431445517883817,-0.9157979919230278,-2.482872132744675,-2.649732863982952,-1.3723243648175314,0.6460585157360668,1.703107437163888,1.778753218422044,1.760030050834977,1.1815004537779807,0.865849308758885,0.5926740829213335,0.6786093234336094,1.2723005844277733,2.8136810842200752,5.523751606256782,8.0450637179879,8.717482621299904,7.809338677032391,5.544562842549258,1.6881159457167023,-2.5300162927446097,-5.589108984461955,-7.005785888410455,-7.4076511432592405,-7.754951885899904,-6.528144371936633,-4.612329974576564,-1.043812670663741,3.3044873559547776,5.733493710906458,7.46024340303177,7.472291660492738,7.161440611880599,6.247880204292571,3.738204636790394,-0.17564428963310613,-4.391658143582436,-7.2281758051553116,-8.537650032130301,-8.518041326731721,-6.4918973414575305,-3.6422332793370913,-1.626168218069138,-0.949720673045915,-0.573759266705911,-0.756864553436445,-1.1327678498279259,-1.556635093009218,-1.7692122630502833,-1.793984358547406,-1.110852161882649,0.6977983667501919,2.389808051553775,2.711453936679473,1.5702876804482893,-0.5609305266577564,-2.391455085114814,-3.3469621767990354,-3.2312989717472425,-2.1553426331848726,-1.0303727713281305,-0.07729922551463472,0.0015299714382827926]
    p2      = 0.0
    # B20QI_deviation_max = 0.20109253151536
    # B2cQI_deviation_max = 0.5349674870675338
    # B2sQI_deviation_max = 0.08280780115372033
    # Max |X20| = 8.361730511239745
    # Max |Y20| = 11.518704078442351
    # gradgradB inverse length: 8.946798979031273
    # d2_volume_d_psi2 = 205.85690952424648
    # max curvature_d(0) = 3.8121944359494564
    # max d_d(0) = 1.978347778106667
    # max gradB inverse length: 3.814902342435955
    # Max elongation = 5.0037049067668296
    # Initial objective = 61.74610613849909
    # Final objective = 61.70844180496931
    return Qic(omn_method = omn_method, delta=delta, p_buffer=p_buffer, p2=p2, k_buffer=k_buffer, rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, B2c_cvals=X2c_cvals, B2s_svals=X2s_svals, order=order, d_over_curvature_cvals=d_over_curvature_cvals, B2c_svals=X2c_svals, B2s_cvals=X2s_cvals)