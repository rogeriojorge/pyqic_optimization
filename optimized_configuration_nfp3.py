from qic import Qic
def optimized_configuration_nfp3(nphi=121,order = "r3"):
    rc      = [1.0,0.0,-0.027765915928235387,0.0,0.0005512008562445916,0.0,-0.00016180072249463538,0.0,0.0]
    zs      = [0.0,0.0,-0.03441507197697066,0.0,8.941178215488026e-07,0.0,-0.0002902931192998124,0.0,-7.73085892255292e-06]
    B0_vals = [1.0,0.18729638313640679]
    omn_method = "non-zone-fourier"
    k_buffer = 1
    p_buffer = 2
    d_over_curvature_cvals = []
    d_over_curvature_spline = [0.7584321499595021,0.5438808607306307,0.395555008529884,0.4022210498005707,0.4604909490761839,0.5067639037223595]
    delta   = 0.1
    d_svals = []
    nfp     = 3
    iota    = -1.7159086623402806
    p2    = 0.0
    X2s_svals = []
    X2c_cvals = [0.0]
    X2s_cvals = [0.0021405751128522844,0.036317746968101555,0.10546568609635061,0.19709735801718203,0.29477970667124775,0.38109528609787985,0.4404144423495418,0.46075466107734586,0.4342994433128482,0.3566200691766027,0.22518017458223286,0.0381005219781857,-0.20584348724740215,-0.5056318294267151,-0.8551789364646056,-1.2404492950658048,-1.6386866975197807,-2.0210928109803454,-2.3585064312480157,-2.628030624786157,-2.818034381777424,-2.9297648099889892,-2.9753617354970427,-2.9734339453307945,-2.943921776185442,-2.903709803860632,-2.864118496934899,-2.830687886376733,-2.804214463796544,-2.7822111006763675,-2.7603556854304245,-2.7337015558563147,-2.6975752071554377,-2.648182635389236,-2.5829832456818815,-2.5008981330657116,-2.40240779030807,-2.2895751501038046,-2.1658892572059933,-2.035784134358712,-1.9040718760800832,-1.775415129537932,-1.6538120498194386,-1.5420424899109764,-1.4410702868930836,-1.3495429716906455,-1.2637118828577254,-1.1781159382908248,-1.0870631666456956,-0.9864220349371486,-0.8749320404551606,-0.7544728663109439,-0.6293020166992737,-0.5047962193576379,-0.3863152771104314,-0.2785181588326937,-0.1851275712145197,-0.10896780642982762,-0.05209786280121796,-0.015935684034787394,-0.0013392136692661428,-0.008646813361333959,-0.0376895886618372,-0.08778560870288472,-0.15771037335858143,-0.24563044239038023,-0.34899232305235034,-0.4643847640237495,-0.5874524106639334,-0.7130178249961773,-0.8356028522020702,-0.950432261067528,-1.0547104345666836,-1.1486236800055931,-1.2354475399168314,-1.32054545442965,-1.4096503554010344,-1.5072011389648245,-1.6153722268960449,-1.7339290227196174,-1.8606306309840774,-1.9918203812895077,-2.1229990762297595,-2.249347273697305,-2.3662440659689477,-2.4698213697796816,-2.557506562163988,-2.6282905075702314,-2.6827388615839345,-2.7229218131061095,-2.7522475346216004,-2.7751509291342913,-2.796571913999074,-2.8211602639963425,-2.8521667547564387,-2.89005821064402,-2.931022692234144,-2.965730268113293,-2.9789947710198286,-2.9514762597216824,-2.863564083554058,-2.700399107194516,-2.4567292147930084,-2.1398826109959885,-1.7694120133644207,-1.3730968470139127,-0.9805876225910768,-0.617127601610782,-0.2997327328804578,-0.036877161140674374,0.16908525979223757,0.31887958067473277,0.41424690100508643,0.45741110883175184,0.45199460924737306,0.4045458249642094,0.32564615168518835,0.22985477379025665,0.13426913735065876,0.055986705244419115,0.009117447556459973]
    X2c_svals = [-0.31036650030642776,-1.321067515709374,-2.2090662231264893,-2.923968443612366,-3.4170141776875456,-3.6595567553600867,-3.6438917372185498,-3.3794775785963,-2.885808408621973,-2.1837875970052223,-1.288415346590275,-0.2060129941324823,1.0617372862583034,2.507502247104659,4.104528512009012,5.794573022069561,7.486192757274541,9.06740658907683,10.429426641506971,11.492521441197583,12.224028842608009,12.64243544634881,12.807665952991599,12.802697098589254,12.71326503505837,12.610998182961058,12.544027944731521,12.53654644354115,12.592991675940716,12.703612223649834,12.849864112543727,13.008904130047998,13.157003473261998,13.272043076175835,13.33536619431273,13.333276808822628,13.258416792186704,13.11117309136757,12.90045942051878,12.642740069626434,12.360246083647331,12.078917198972894,11.825754951613797,11.624976040219146,11.492446539230919,11.428846728759071,11.413862534614378,11.40517488030848,11.345135959901201,11.174152292004047,10.845477255881189,10.335015305366568,9.642501433916003,8.785465829973322,7.7905122182897255,6.686033841425578,5.498240758231718,4.250488640399441,2.9660306921581574,1.6834234819300746,0.9899604174901996,-1.2785876798370144,-2.5349946427745857,-3.8251363922395973,-5.087886037987497,-6.2981572774817325,-7.433163549071752,-8.467720127496646,-9.373893790521533,-10.123823453672669,-10.69599000511748,-11.08382124613282,-11.303168457637348,-11.393920336374043,-11.412905964826823,-11.42011589969837,-11.464120581427975,-11.572803104133706,-11.751990035495949,-11.990101324623954,-12.264998433353366,-12.550101538865427,-12.818778497939185,-13.047313340458087,-13.217111253773943,-13.316563133896787,-13.342444638058684,-13.299688909921061,-13.2000311796788,-13.060631583123117,-12.902637455803616,-12.749480699238553,-12.624628276905563,-12.548502928219877,-12.534349936388777,-12.583095775604466,-12.67770550065593,-12.778312376087655,-12.82055930124925,-12.721819529955559,-12.396026827292742,-11.773298268607046,-10.819691266192379,-9.550527603866866,-8.031230414284384,-6.363573630396024,-4.661434286237286,-3.0251897884528276,-1.5246602525175525,-0.19609932849245626,0.9482947630869374,1.906330123998331,2.674037227564916,3.2393081914712334,3.5825550168589255,3.6828538812052525,3.5264876576423823,3.1148268964785877,2.469374193604139,1.6326442060829132,0.6614429797315574]
    # B20QI_deviation_max = 0.004956494901298569
    # B2cQI_deviation_max = 1.9467985001853094e-09
    # B2sQI_deviation_max = 0.00014615747530033474
    # Max |X20| = 12.824338384121836
    # Max |Y20| = 4.581149863961276
    # Max |Y2c| = 2.909067549519345
    # Max |Y2s| = 4.842342704970193
    # Min r_singularity = 0.048815966734332285
    # Max |X3c1| = 1.8149973275817184
    # Max |X3s1| = 20.537322514341696
    # Max |Y3c1| = 136.0735215676024
    # Max |Y3s1| = 8.401478560009513
    # gradgradB inverse length: 7.640099316847599
    # d2_volume_d_psi2 = -246.9039091814715
    # max curvature_d(0) = 8.041515970703207
    # max d_d(0) = 6.0933067124392615
    # max gradB inverse length: 3.9009449508565863
    # Max elongation = 6.69977598155653
    # Initial objective = 48.74171150067447
    # Final objective = 48.65765546758451
    stel = Qic(omn_method = omn_method, delta=delta, p_buffer=p_buffer, p2=p2, k_buffer=k_buffer, rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, B2c_cvals=X2c_cvals, B2s_svals=X2s_svals, order=order, d_over_curvature_cvals=d_over_curvature_cvals, B2c_svals=X2c_svals, B2s_cvals=X2s_cvals, d_over_curvature_spline=d_over_curvature_spline)
    stel._set_names();stel.calculate()
    return stel