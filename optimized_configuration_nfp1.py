from qic import Qic
def optimized_configuration_nfp1(nphi=131,order = "r2"):
    rc      = [1.0,0.0,-0.3217551889242487,0.0,0.03482662049168943,0.0,0.00045198368277089954]
    zs      = [0.0,0.0,-0.22503006227732608,0.0,0.03727671683847557,0.0,-0.003726813808018676]
    B0_vals = [1.0,0.21033116699038865]
    omn_method = "non-zone-fourier"
    k_buffer = 1
    p_buffer = 2
    d_over_curvature_cvals = [0.45104854187633714,-0.019654612770548106,0.002113655780420127,-0.0006292835289798944,-0.0043644987759675164,4.0170139513142185e-05]
    delta   = 0.1
    d_svals = []
    nfp     = 1
    iota    = -0.6958969671972872
    X2s_svals = []
    X2c_cvals = [0.0]
    X2s_cvals = [0.01252543658186715,0.1721338462428749,0.4194952579309511,0.568388476194862,0.4262757453914187,-0.14313605448183028,-1.1703780698557569,-2.5325060221957987,-3.9313959924847874,-4.9674658169497405,-5.332947902942305,-4.994724401938431,-4.186677454407553,-3.227110586679651,-2.3488789312470235,-1.652537099150283,-1.14515955997935,-0.7928685224142453,-0.5541723853781627,-0.3937883052315854,-0.28563056630554307,-0.21164763378724766,-0.15977493804572745,-0.12216223447145155,-0.09390281852420071,-0.0721773884707828,-0.05566080821096392,-0.04406301880287962,-0.03772242153520411,-0.037221945507070954,-0.04304023743931868,-0.055275402024347456,-0.07347895448762612,-0.09661885208182529,-0.12316129878270994,-0.15123654078647789,-0.17884022289627163,-0.20402338137637793,-0.22503623432472397,-0.24041187880621318,-0.24899836918831503,-0.2499663409103537,-0.2428240470142582,-0.22745996425382595,-0.20420674338793712,-0.1738938192658551,-0.13784318345119131,-0.09777543240530064,-0.05562681041379916,-0.013319954666728962,0.027438428323966244,0.06524946708153466,0.098928533409418,0.12716158725093188,0.1478443198063614,0.15701163145732322,0.14723265673240313,0.10536495448654949,0.009791644347381792,-0.17194701392640652,-0.4823120330274498,-0.9636495034079771,-1.6304133019156937,-2.4224725771110918,-3.166277044610684,-3.605138388722254,-3.532353143365716,-2.9400672066658555,-2.034690335603453,-1.1042009142285296,-0.372080981801996,0.06318260882760525,0.21583396167686913,0.17576923839442002,0.06731451046667403,0.002805556061427231,0.03667229286270755,0.14094650929300856,0.21804283507762107,0.14220246301718809,-0.19271900501491757,-0.8306011177946206,-1.7139506474629242,-2.658857187495618,-3.386012320676519,-3.6431556986785503,-3.3593774592347345,-2.687611932728223,-1.8865996978249093,-1.1662288288403466,-0.6220884302373939,-0.2588419201010577,-0.039281876267366414,0.08083990869602124,0.1377075755183301,0.15636352964661726,0.15245651219482081,0.1350394251429719,0.10902401652695988,0.07699232120673363,0.040430391044164245,0.0005136702493252467,-0.041451862340169086,-0.08385964681267966,-0.12484029989105173,-0.16243627246709666,-0.19483121602512632,-0.22055501605482178,-0.23860957448252873,-0.24850061584751318,-0.25019903500565516,-0.24407603921404566,-0.2308512823418521,-0.21156977920510273,-0.18759512723587443,-0.16058917615453525,-0.1324475729033782,-0.10517557801450327,-0.08071051610664612,-0.06071938645502134,-0.04641562828425399,-0.038444487674820414,-0.036877862748626394,-0.041338035496947284,-0.051240791478878354,-0.06612470029188602,-0.08602642779579966,-0.11187883177697618,-0.14594611759973566,-0.19236012555330087,-0.25787221694671053,-0.3529699975301946,-0.49348432553084476,-0.7026093389343295,-1.0125885428815313,-1.4636660337584722,-2.094838499194985,-2.9178188621473096,-3.869403637103207,-4.76151772100691,-5.291000385394195,-5.172061174968269,-4.336394664094598,-3.0138091156164197,-1.598471340609037,-0.4371586982139952,0.2880946135651753,0.5617385944714492,0.4901308719226427,0.2530887342237091,0.04702010445192774]
    X2c_svals = [-1.0033548553830862,-3.4462160165565883,-4.796121566122267,-4.544483759443573,-2.618324942803917,0.7132002986693529,4.901709485115804,9.141300319473254,12.460701016858042,14.027149048077156,13.574437112197113,11.572284676058107,8.903575041874388,6.346058998517242,4.299811162557308,2.8334898755745637,1.849860145268829,1.2133084719498128,0.8076090131910666,0.5492848517745681,0.38319108938013285,0.27434776755220797,0.20099212347335588,0.1497620338070046,0.11263285292488041,0.08502872291648854,0.06463912748258782,0.05062590103830168,0.043033799392380376,0.042314148445938504,0.04893514103994249,0.06308780372854303,0.08450340980835848,0.11238671169517982,0.1454491288753853,0.1820106588136479,0.22013183167801145,0.25774022900704846,0.2927250975093177,0.32298967856538907,0.3464701740791324,0.3611512698149578,0.36511909750954435,0.3566858915778246,0.3345896509082804,0.298228431950306,0.24785042469761787,0.1846140685508858,0.1104669696654255,0.027864652441911747,-0.060565083286934634,-0.15222762118625496,-0.24438457736849206,-0.33325409373047915,-0.4118886184705449,-0.4660312065442957,-0.4666814500475753,-0.3575647134811796,-0.03567494806351212,0.6746478265189656,2.044827612065663,4.431476438777136,8.168012802868372,13.287953735893318,19.1300243050312,24.1638615860839,26.497192263642205,24.95519042210899,19.819891924251724,12.576508017794062,5.0840277150034145,-1.0749631403589837,-4.847331240219912,-5.782880721605295,-4.118895222670773,-1.2023699872076001,3.14240944639903,5.483488752870398,5.477830050581046,2.632851695462825,-2.819067704202388,-10.018998980620566,-17.547456127054925,-23.586777868303148,-26.431734297957984,-25.325282066577657,-20.993039783391477,-15.214044401316848,-9.735639851714353,-5.51551909303837,-2.708653367968034,-1.0417435507137085,-0.14664627397420776,0.280819891393214,0.446403097038924,0.47425500415911503,0.4336918335267162,0.3611023324450775,0.27462187578246533,0.1830430658876555,0.09088759772348701,0.0010937695810694134,-0.08374313018808416,-0.16099656554103053,-0.22811825108600392,-0.2829543537316671,-0.3240607659787191,-0.35088410021736405,-0.3637592384157419,-0.36375387710678386,-0.35243902901852797,-0.33167107925649936,-0.30344044627463707,-0.2697974680024982,-0.23282906984040525,-0.1946451115519616,-0.15733925616627986,-0.12290818627496243,-0.09313345585866564,-0.06944772722725397,-0.052818532070409055,-0.043688127145297295,-0.042004269146391174,-0.04736370863222825,-0.05927056796113016,-0.0774959918691014,-0.10252505329273284,-0.13610220500715073,-0.18194109071658257,-0.24674984394958874,-0.3418404387031549,-0.4857389245858711,-0.7083538196148129,-1.057214831242464,-1.6055027371672537,-2.458706218121454,-3.7495346864111245,-5.598493225887577,-8.010595033321074,-10.711961131184925,-13.038875011908633,-14.0914282512049,-13.20607291501095,-10.398161027529255,-6.35564567616195,-2.0488730284152186,1.642350896531133,4.081393640790015,4.901835370475575,4.052881168830459,1.8831727619348029]
    p2      = 0.0
    # B20QI_deviation_max = 0.000801725157722144
    # B2cQI_deviation_max = 0.35168960957014245
    # B2sQI_deviation_max = 0.00122080945349623
    # Max |X20| = 25.004327160817574
    # Max |Y20| = 31.734154087687426
    # Max |X3c1| = 14.640623878497774
    # gradgradB inverse length: 9.145266409844393
    # d2_volume_d_psi2 = -148.79048418865165
    # max curvature_d(0) = 1.5211591150328279
    # max d_d(0) = 0.6519256301482697
    # max gradB inverse length: 2.5845622789379474
    # Max elongation = 6.021807956576866
    # Initial objective = 40.6626693114592
    # Final objective = 44.47895453129075
    return Qic(omn_method = omn_method, delta=delta, p_buffer=p_buffer, p2=p2, k_buffer=k_buffer, rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, B2c_cvals=X2c_cvals, B2s_svals=X2s_svals, order=order, d_over_curvature_cvals=d_over_curvature_cvals, B2c_svals=X2c_svals, B2s_cvals=X2s_cvals)