from qic import Qic
def optimized_configuration_nfp3(nphi=61,order = "r3"):
    rc      = [1.0,0.0,-0.02959740086970366,0.0,0.000701476191461077,0.0,-2.0339124870218257e-05,0.0,0.0]
    zs      = [0.0,0.0,-0.02930418651568733,0.0,-0.0002539525417201953,0.0,6.0568370156111146e-05,0.0,1.1656640923561373e-05]
    B0_vals = [1.0,0.1883644672999329]
    omn_method = "non-zone-fourier"
    k_buffer = 1
    p_buffer = 2
    d_over_curvature_cvals = []
    d_over_curvature_spline = [0.5613521753424431,0.4890751386373724,0.3993466968057703,0.5002909159272995,0.5012870789802362,0.5283193077595487]
    delta   = 0.1
    d_svals = []
    nfp     = 3
    iota    = -1.7235246130012685
    p2    = 0.0
    X2s_svals = []
    X2c_cvals = [0.0]
    X2s_cvals = [0.00015960170477834085,0.05857384032220911,0.16347248291515917,0.2675803570217434,0.2992566270028044,0.15718855347419214,-0.27576987089028704,-1.073155609058668,-2.175845619430441,-3.3534502552179997,-4.248925095903452,-4.491851750543652,-3.9261560602047205,-2.7986563928011465,-1.5868005274910177,-0.6379549977423059,-0.050895038667684134,0.2081856402249962,0.17938653278129693,-0.12322685569922413,-0.6595269367909474,-1.293626399556726,-1.8093154733781036,-2.0407179812925778,-1.9599131919621493,-1.639062014566175,-1.1917981711536918,-0.7343670750415231,-0.3564231894368267,-0.1079204232555071,-0.006981790502083953,-0.05753607435966697,-0.2578290617701185,-0.5962600428758007,-1.0352622969042604,-1.4980050923947428,-1.8744628024278658,-2.0462796189678536,-1.9222699102881025,-1.489360050320597,-0.8699924086736389,-0.28032036379757935,0.10826809727246671,0.22957852262239237,0.06929638548117546,-0.402542142237845,-1.231917670447329,-2.381609959363251,-3.590143466300123,-4.389471292701245,-4.417284283501209,-3.702925003362773,-2.575828500017567,-1.4147842784125952,-0.5005035016632853,0.05057886086261029,0.2769987877885214,0.2905714227110332,0.20123014076651785,0.09027358703087383,0.012429372317922936]
    X2c_svals = [-0.012059318632759497,-1.121312940446533,-1.8405472466446298,-2.2016861994610304,-2.002047806415739,-0.9121997453740405,1.4494541774755114,5.265007247017164,10.182598763487865,15.214788355399834,18.93383180475392,19.883854432931432,17.451214274192512,12.624664822722655,7.342620056431158,3.0605275012866713,0.2557858828923315,-1.1071609920276702,-1.019354369210852,0.7553835271733861,4.404297323574759,9.511633320979323,14.829075896116706,18.928821363990554,20.982986125012882,20.805066480752703,18.64265981979216,15.03391800376992,10.61958604109878,5.955105281683593,2.6937493428129153,-4.442303918876504,-9.068787969974874,-13.62157598464399,-17.5688309082672,-20.282606347011853,-21.16708650271532,-19.860711797491124,-16.38765193110731,-11.33757471638923,-5.99173767314207,-1.7661446033608705,0.6303030270505581,1.2468673651962086,0.3545040454656718,-1.9590893546260992,-5.762706961911005,-10.822078568083604,-16.017397453213555,-19.434246038341534,-19.616678054585574,-16.67762691506857,-11.911469258495817,-6.8184367749560915,-2.5635986945131033,0.2828457808063671,1.756986052991682,2.2112406277918897,2.008096938354759,1.3935421665753558,0.47083573862744127]
    # B20QI_deviation_max = 0.013214623887451982
    # B2cQI_deviation_max = 1.891483940585914e-07
    # B2sQI_deviation_max = 0.005466679685635967
    # Max |X20| = 18.774361029729388
    # Max |Y20| = 18.7206118607427
    # Max |Y2c| = 11.43982011328845
    # Max |Y2s| = 12.17252093897965
    # Min r_singularity = 0.020120520149167546
    # Max |X3c1| = 8.400437646183232
    # Max |X3s1| = 104.63949273445604
    # Max |Y3c1| = 491.48256727171463
    # Max |Y3s1| = 10.125056929465584
    # gradgradB inverse length: 9.738493290178509
    # d2_volume_d_psi2 = -108.76918686306647
    # max curvature_d(0) = 5.6696511211363605
    # max d_d(0) = 3.179262869345087
    # max gradB inverse length: 4.13346885255766
    # Max elongation = 6.0864503032301736
    # Initial objective = 2309.7786947427517
    # Final objective = 50.25000315652732
    stel = Qic(omn_method = omn_method, delta=delta, p_buffer=p_buffer, p2=p2, k_buffer=k_buffer, rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, B2c_cvals=X2c_cvals, B2s_svals=X2s_svals, order=order, d_over_curvature_cvals=d_over_curvature_cvals, B2c_svals=X2c_svals, B2s_cvals=X2s_cvals, d_over_curvature_spline=d_over_curvature_spline)
    stel._set_names();stel.calculate()
    return stel