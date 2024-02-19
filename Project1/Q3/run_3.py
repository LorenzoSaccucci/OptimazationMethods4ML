#Group the Treesss
#import needed libraries
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from function_3 import *

Data=np.array([[-0.0168156303388702, 	1.49361612676431, 	-0.996793759488599],
                [-0.0954450458752047, 	2.74278733075779, 	-1.16915490063471],
                [-1.90014211720466, 	0.674755716264015, 	0.256156846840682],
                [-0.982228844645768, 	1.46448304398304, 	0.880567437955377],
                [0.919084459473148, 	1.49532169937352, 	-0.863522815479265],
                [-1.06740702507766, 	1.58266812855955, 	1.46630108867027],
                [-1.98190522195036, 	-0.727572189500239, 	-1.64370083203503],
                [0.607004658631801, 	1.75515471158006, 	-0.916813025803028],
                [1.27023399301225, 	0.593176712878097, 	-0.846374446662334],
                [-1.32424365990288, 	2.01166491017186, 	3.33286164037268],
                [1.24797981517295, 	0.079772519326541, 	-0.749105581855029],
                [0.881936832821172, 	1.17980192935222, 	-0.940206993023232],
                [0.921811079685452, 	2.19912631031561, 	-1.05706401502072],
                [-1.96801001116406, 	1.86164369618324, 	4.29320587858214],
                [-1.62786015619294, 	1.0887743669193, 	1.81597481292154],
                [-1.53622509120932, 	0.761140747529377, 	0.414221719490376],
                [-0.700634705842319, 	2.16646539511003, 	0.42659213193345],
                [-0.405942800067009, 	-1.76274110454373, 	-0.375900542964002],
                [-0.485903346769792, 	-1.19877493502079, 	-0.901269074439316],
                [-0.101563689671516, 	-2.53595702158953, 	-0.152700538965433],
                [-1.92522559531621, 	1.23841608911641, 	2.59430982203285],
                [-0.467857623662467, 	0.0673234125281028, 	-2.89733772193386],
                [0.118221774302809, 	-1.19074108739829, 	-0.7262526441979],
                [-1.59436925598616, 	-2.27469725718979, 	-1.14146482833082],
                [-1.25412129350704, 	1.17475491342784, 	1.14805911680775],
                [1.90437420297278, 	-1.81018789732127, 	3.30583096149492],
                [-0.209456241331227, 	2.31986077488738, 	-0.796093818358824],
                [1.40727829257545, 	1.24927345797762, 	-0.803548631325091],
                [-1.06215885803872, 	-1.52718060223268, 	-0.779983507405245],
                [-1.18198588942201, 	-2.45566872880589, 	-0.783566929543371],
                [0.462883213535146, 	0.882908650471636, 	-1.60371190620875],
                [-0.298170061236314, 	-2.91216566067721, 	-0.273059644532715],
                [-0.118075375000472, 	-1.52004925360206, 	-0.411639456500278],
                [-0.869902887544642, 	0.458454497921588, 	-1.62073572978234],
                [-1.24305407090709, 	0.941413905758177, 	0.394782965891098],
                [-0.487644386320928, 	1.01099007239137, 	-1.27059598695206],
                [1.75961197360236, 	0.122852402636013, 	-0.347913922183977],
                [1.94072881937556, 	2.42165815761321, 	-1.46522184637646],
                [0.230626541441318, 	1.08261848665329, 	-1.45685497716854],
                [-1.34280283819363, 	0.57201759556189, 	-0.447406169536488],
                [1.01562810089292, 	1.86383126126826, 	-0.930341719378916],
                [-1.50507579721706, 	-2.73311748729205, 	-1.0841827049407],
                [-0.0388161037036156, 	2.04738756534825, 	-0.871027969457585],
                [-0.317923780189776, 	1.57443988509569, 	-0.719081864768884],
                [0.589415726184918, 	1.17546911089646, 	-1.12195901385396],
                [-1.25280447048865, 	2.50811163029078, 	2.62461875685307],
                [-0.515641355935877, 	-2.84658596929672, 	-0.371476247151898],
                [-1.23358566962571, 	2.05984111198402, 	2.92082101649701],
                [0.606014376616271, 	-1.46935721372994, 	0.31430622849754],
                [1.61774151309593, 	0.771090476519551, 	-0.717285844618584],
                [-1.8071269305776, 	2.36746828768891, 	4.24108576651264],
                [0.0519056922863847, 	-0.239416425031926, 	-3.08017273775413],
                [1.78677892250848, 	-0.557928089302208, 	1.12914212196519],
                [1.65259668450162, 	1.3424184669085, 	-0.868200074819526],
                [-0.156295376437171, 	-1.7519446755107, 	-0.267896935958599],
                [1.48359307426859, 	2.86681788182925, 	-1.52292370397475],
                [-1.58477599761988, 	1.34216769129797, 	2.72442060891247],
                [-0.687857448116521, 	1.09876701444583, 	-0.719759812709691],
                [1.36091242211522, 	-0.953476739098906, 	1.63701035202628],
                [-0.196686594959521, 	0.838535918307579, 	-1.95564773292649],
                [-1.59193909735982, 	2.17307054833986, 	4.34149063951367],
                [-0.977217219021344, 	0.282760287996373, 	-1.64342162426364],
                [0.126567140879484, 	0.121453361341633, 	-3.23240831946148],
                [-1.22248489821949, 	-0.196482846116149, 	-1.5113851664326],
                [1.49843990686169, 	-0.741857208478994, 	1.35286134743948],
                [-0.522899995898243, 	2.97771324487382, 	-0.907656545762656],
                [1.60512175581244, 	0.496025347617637, 	-0.640389068977241],
                [1.9231130017641, 	-2.06908364938041, 	2.57580748814545],
                [-1.19486969774673, 	1.35811908072452, 	1.46884317381828],
                [0.440348665496358, 	1.52123553988078, 	-0.968325786867466],
                [-1.87719509977999, 	0.324666746005323, 	-0.789609686794887],
                [1.02890995732425, 	-1.51602342831935, 	1.42175698262393],
                [-0.24921266935977, 	-1.08400073470945, 	-1.09747017846869],
                [1.05457639779903, 	-1.61356479225019, 	1.51262249915632],
                [-1.86738923020172, 	0.880661784384973, 	1.09134340447364],
                [1.77374311965001, 	-2.27339268237275, 	1.85455436860938],
                [0.0881616412028059, 	2.70736519805951, 	-1.23760729780508],
                [-0.761274705496697, 	-1.25653012448654, 	-0.841814073242421],
                [0.0448688937814334, 	2.62498495464893, 	-1.16852276689501],
                [0.0086467947546258, 	-0.700536225458018, 	-2.00174065711202],
                [1.18795038449606, 	-1.11334165754967, 	1.4814725571748],
                [1.39953504756582, 	-0.659540845230131, 	0.905201100112268],
                [1.88848658676338, 	2.6263671353001, 	-1.54951218188676],
                [1.78529536825623, 	-1.63108795053177, 	3.50529724657212],
                [-1.44774745737781, 	-1.81004263735492, 	-1.01133267677569],
                [-0.623928230434067, 	1.23954689330062, 	-0.603172786977162],
                [-1.15518388501786, 	-0.3912251509823, 	-1.51802703481697],
                [-1.19494041003648, 	-1.25321980039103, 	-0.95734404043342],
                [1.6034803134794, 	2.12862042318719, 	-1.17055209948445],
                [1.3087524890107, 	0.981586334754155, 	-0.794154335022283],
                [-0.510318285639279, 	-1.98245817409641, 	-0.362671893915573],
                [-0.870410758810756, 	0.176648926273582, 	-1.9735681571506],
                [-0.94763035821305, 	0.463733193697109, 	-1.44027559579955],
                [1.06821313327401, 	-1.03687405292082, 	0.951366144718925],
                [-0.93185396357104, 	0.167082426089559, 	-1.85124671822632],
                [-1.55341883823208, 	-2.77288650437443, 	-1.13373829297336],
                [1.34767634637904, 	-1.57379170395705, 	2.46898167838674],
                [0.106175102706237, 	-1.33955953542207, 	-0.490679704125836],
                [1.74138524466653, 	-1.80486202301778, 	3.18717060232621],
                [1.77083786666234, 	-2.77099968414948, 	0.488470850149442],
                [-1.32462830191829, 	2.08095367868784, 	3.37087233937248],
                [0.5175190856373, 	1.31025292534142, 	-1.05392751175697],
                [-1.46089438385563, 	-2.76172254508552, 	-1.04578105917286],
                [-1.01634534825116, 	1.12853696488749, 	0.24073139864441],
                [0.322897737815045, 	2.64657735951866, 	-1.24987066623724],
                [-0.131004380373179, 	0.595892310840158, 	-2.55494906971311],
                [-0.502824026844947, 	0.839897697206117, 	-1.60476098249963],
                [-0.20911740719559, 	-2.3948781961939, 	-0.191672293025419],
                [-1.66833142973816, 	1.94888111829283, 	4.44820643720335],
                [0.794271594805968, 	0.319738316695914, 	-1.7221461642889],
                [-0.503158642733394, 	-2.35126002065874, 	-0.333633111260438],
                [0.437068743338027, 	1.67898987803084, 	-0.935049077680001],
                [0.213890468262157, 	0.580761938472101, 	-2.47428905405194],
                [0.672312908680602, 	-1.2291818168338, 	0.182774592663535],
                [0.665480226658554, 	0.972093434448221, 	-1.26381264227925],
                [-1.01722413212985, 	1.30756846769889, 	0.673268766931981],
                [0.662740641667698, 	-1.09084912438374, 	-0.069712660426791],
                [0.824769592801124, 	-0.848758013386719, 	-0.158920829082722],
                [-1.89382610490959, 	0.0225962309622502, 	-1.31048090787681],
                [1.89714686664361, 	-1.0190443596105, 	2.77117799622981],
                [1.6627400564238, 	-1.27306970033344, 	3.1415385546886],
                [-0.719478553689381, 	-2.84188303555524, 	-0.486968271959814],
                [-0.135184817261712, 	-0.737660313214632, 	-1.93354050513792],
                [-1.16252253243683, 	1.56227062010157, 	1.84241338014695],
                [0.0433011464873591, 	-2.43315801204577, 	-0.0835349943951532],
                [1.15599085043674, 	-2.37429433810536, 	0.86279771567695],
                [-0.260271764703405, 	0.0002920351903963, 	-3.21837228064344],
                [0.71553542383709, 	0.855586701411158, 	-1.3384092668875],
                [-1.8556254683393, 	-1.52842922955957, 	-1.42450765413821],
                [1.80711654799394, 	0.719001665983833, 	-0.71774907680049],
                [0.582955997379086, 	0.46999565925688, 	-2.06097267867296],
                [0.798567931786958, 	1.51694823980217, 	-0.88501033851873],
                [0.0126832907378271, 	1.35985796085502, 	-1.12025611340673],
                [0.650326686844447, 	1.67389170685034, 	-0.906595743671069],
                [0.471161370185007, 	0.945278876978668, 	-1.49674455357174],
                [1.02914447945576, 	2.64266639591175, 	-1.29430781872964],
                [1.0307397937634, 	1.63288541450658, 	-0.868497114064223],
                [0.707596908380824, 	1.65288376571705, 	-0.898112659963126],
                [1.9096451130176, 	-2.17791105858864, 	2.21019391327903],
                [-0.314593816728711, 	-2.63340433956304, 	-0.252980155379274],
                [-1.84695568999333, 	0.0923694081620229, 	-1.19408595977978],
                [0.282965311481092, 	2.2254195136111, 	-1.03622200175417],
                [-1.54713712126793, 	-1.7186932541042, 	-1.10687668929116],
                [-1.11951575042274, 	-1.08918841188677, 	-1.02400547965364],
                [-0.411187723598936, 	-0.169003895240722, 	-2.90766479538327],
                [-1.96539905267723, 	-0.519945256930479, 	-1.63188021227672],
                [0.463589123311634, 	-1.14736320447478, 	-0.370894256778484],
                [-1.26783614922172, 	0.422220685243539, 	-0.900332499221209],
                [-0.858875043135764, 	2.3585580599661, 	0.997388448084025],
                [1.90780528043241, 	2.93029625726429, 	-1.73210053075339],
                [0.938107639629958, 	-0.914436151669059, 	0.304073782403952],
                [-1.32586087057966, 	0.887629983141859, 	0.437329612977495],
                [1.59673003023966, 	-0.861782346711316, 	1.91925904191437],
                [1.72092873830665, 	-2.10138606742886, 	2.36819686859401],
                [1.67246669671303, 	-2.65713297508315, 	0.726803347967778],
                [-0.0329922228370707, 	-1.53635465880372, 	-0.352320593903925],
                [-1.41583611359388, 	-0.587054280054107, 	-1.33122625054153],
                [-1.93139598848401, 	-1.92532329526063, 	-1.48929001870018],
                [-1.05460131078801, 	0.989367074006808, 	0.0018105573998293],
                [-0.45722611485535, 	2.22807956645674, 	-0.32386051877434],
                [0.221375212626111, 	2.21908402689452, 	-1.02324093860561],
                [1.78635746163454, 	0.597875497927364, 	-0.669408050729503],
                [-0.0579894247275465, 	-2.06467547603526, 	-0.130942758231177],
                [-1.9111277349895, 	-0.54363972231215, 	-1.58342664219557],
                [1.42454193694679, 	1.68085513718384, 	-0.924162004357976],
                [-1.6202296048559, 	0.280007264915032, 	-0.847539794166986],
                [-0.173059489699982, 	-0.872856830786979, 	-1.57800267332333],
                [0.600221876030448, 	1.24484146434905, 	-1.05953761713455],
                [-0.817191762367542, 	1.62049234675568, 	0.516161728855112],
                [0.230241767432492, 	-2.90144650159978, 	-0.0806505419955262],
                [-0.284723241622881, 	-1.60106478799194, 	-0.416099382948821],
                [0.751999350388934, 	-2.52082500265704, 	0.275946920583681],
                [-0.0414330759938268, 	1.21105794426798, 	-1.29572993106197],
                [-1.88768777590243, 	2.88502812673793, 	1.97857466084463],
                [1.71231856813112, 	-2.50938345299694, 	1.10894501244463],
                [1.45263435604797, 	2.40274786274956, 	-1.2578829823384],
                [-1.16821714225731, 	1.68727286465665, 	2.12804262908351],
                [1.84565311156101, 	-0.121381680559167, 	0.0193974489548837],
                [0.401057928647475, 	-1.98541674809748, 	0.172814447143002],
                [0.409251106823722, 	1.26828015200332, 	-1.13765195098321],
                [-0.727993249614648, 	-0.676031985933845, 	-1.65672957136856],
                [-0.993332217575341, 	1.7137596717754, 	1.36403026220502],
                [0.182050294476467, 	-0.807860616232005, 	-1.56408143749346],
                [1.44388279503378, 	-2.62988752971981, 	0.685524791263645],
                [1.41009575631496, 	2.11950474029826, 	-1.10496718056785],
                [1.38574283141493, 	2.07251297031048, 	-1.0765732737473],
                [-1.47246387864086, 	0.180294020501069, 	-1.07656333094715],
                [0.0454195809479247, 	1.19544393987026, 	-1.32732439864507],
                [-1.42594800492316, 	0.881589193207702, 	0.641813634659255],
                [-1.2163672204506, 	-2.99047841019409, 	-0.858110410612095],
                [-0.577767050526262, 	-2.69379625815318, 	-0.390851142886945],
                [1.85041914118532, 	-0.65249591938257, 	1.47561715656755],
                [-1.75982013913096, 	2.27741337791304, 	4.43595263569435],
                [-1.33672023323317, 	-0.949699524774422, 	-1.15688735578616],
                [-0.556308507947007, 	-0.244549397217965, 	-2.57424729764663],
                [-0.0319988858205757, 	-2.43936464607504, 	-0.11631307154515],
                [1.16043414491514, 	0.993746381931495, 	-0.852440974051408],
                [-1.13826672071031, 	-2.5765478761911, 	-0.756070702917447],
                [0.744410167783645, 	-0.13212098443947, 	-1.76912542030816],
                [-0.198012765021504, 	1.98468122658718, 	-0.72848921226439],
                [-1.76047227574289, 	0.601472541161934, 	0.0272195752673199],
                [-1.44384720940104, 	-2.7521142248103, 	-1.0292662573277],
                [0.932537607027638, 	1.02255730650532, 	-0.984681557163904],
                [-0.713423720711892, 	1.20268722732216, 	-0.462246887147768],
                [0.905667013205918, 	-0.0881367951951208, 	-1.36512097489903],
                [-1.6574791412136, 	0.622194996690029, 	0.0606147777926386],
                [1.21832240275207, 	0.133531957427373, 	-0.832908054778853],
                [-0.653085571051452, 	1.82872082416883, 	0.163624357974502],
                [1.04984507873503, 	2.23302229542386, 	-1.08635868197667],
                [-1.176925047944, 	0.648043868468209, 	-0.571150281967698],
                [1.12574832753873, 	-2.17323966291403, 	1.14091981297683],
                [1.65236749989618, 	-0.779048383064863, 	1.73749613730967],
                [-1.56520201051214, 	2.61776135184109, 	3.30838831543152],
                [1.54151905813772, 	1.49452609659738, 	-0.887874375227534],
                [0.364904103908946, 	2.608007969969, 	-1.23221955764725],
                [0.553842546260002, 	-1.6055868057648, 	0.306485954668505],
                [-0.014036062456634, 	-0.991855809203715, 	-1.24147933346767],
                [-1.3973261582239, 	-1.9214507722545, 	-0.961055246509593],
                [-1.25513259247207, 	-1.0035616559382, 	-1.10590234709904],
                [-1.82942846459013, 	2.72895097137693, 	2.88544525015029],
                [0.429155437794057, 	-0.552812169355821, 	-1.83935074783497],
                [1.23418495722682, 	-0.307361636619811, 	-0.277746176472031],
                [-0.406751592990721, 	-0.150762672787116, 	-2.93196849488565],
                [-0.530525063424813, 	2.53925469868043, 	-0.326289944779218],
                [1.32471332962914, 	2.19967259931168, 	-1.12119358703106],
                [-1.60345952185338, 	0.0093744535668394, 	-1.22753633538023],
                [1.94843637921399, 	-2.82020136919835, 	0.343010066924793],
                [-1.87624492634843, 	1.68312491031958, 	4.11362777086254],
                [-1.0283682744361, 	0.815042375770579, 	-0.508775432629182],
                [-1.93308467981871, 	0.252138336572521, 	-0.982398520201369],
                [-0.625535290898426, 	-0.520026010174251, 	-2.04798132597488],
                [-0.340277048463575, 	0.930903272138653, 	-1.62938447100318],
                [1.62456601150789, 	0.569654205761412, 	-0.661993010651297],
                [-1.02579308570254, 	0.972776024066199, 	-0.122027698433437],
                [-0.969877459534015, 	-2.69538867703729, 	-0.639061458157732],
                [-1.26414715358719, 	-2.99675773829221, 	-0.898411006157657],
                [1.63511978874392, 	-1.33697048421192, 	3.17880959157558],
                [0.231087994470335, 	-2.15727589485894, 	0.0321696038791979],
                [-1.10657744842889, 	-0.860804139389304, 	-1.19536215140085],
                [1.09207101240813, 	-1.1179290649501, 	1.18390621720171],
                [0.334575449994197, 	-1.22321147894418, 	-0.435718640300963],
                [1.06613656136764, 	1.8145457756082, 	-0.917534504993414],
                [0.360099090303269, 	-1.32856829570403, 	-0.240624175106336],
                [1.99710004760502, 	2.27903575478296, 	-1.42072468552961],
                [0.148624006902349, 	2.38640673724169, 	-1.07716109383534],
                [-0.968999070863305, 	2.35468250799115, 	1.50983809189178],
                [-1.17729746321828, 	-1.82910545735106, 	-0.791381799143087],
                [-0.498832721586176, 	-1.14012699305874, 	-0.988764839603277],
                [-1.13478301778782, 	0.263514652144472, 	-1.36910106186222],
                [1.39592683363215, 	2.31252048585911, 	-1.19532341718873]])

#dataset definition
X = Data[:, :-1]
y = Data[:, -1].reshape(-1, 1)  #PLEASE PAY ATTENTION TO RESHAPE Y AS (P,1)!

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 186, random_state = 1893639) #dataset division


###################################################
#hyper-parameters definition
print("Hyper-parameters values:")
N = 48 #50 32
print("The number of neurons is: ", N) 
sigma = 1.5 #1 1.4
print("The value of the spread of the activation function is: ", sigma)
rho = 0.00001
print("The values of the regularization term is: ", rho)

#problem dimension
P = X.shape[0] #250
n = X.shape[1] #2
P_train = X_train.shape[0] #186
m = 1


#inizializzazione dei parametri
np.random.seed(4923) #22 4923
w = np.random.randn(N, n)  #initialized with normal distribution
b = np.zeros((N, 1))  #bias initialized to zero
v = np.random.randn(N, m) #initialized with normal distribution
W = np.concatenate((w, b), axis = 1) 

#######################################################
print()
print('-'*100)
print()
print("Optimization part")
print()
#optimization

#definition of the number of iterations for the procedure
iterations = 1000

initial_time = time.time()
r,eval = two_blocks(W.flatten(), v.flatten(), sigma, rho, N, y_train, X_train, iterations)
final_time = time.time()

#Info
print("The running time is:", final_time-initial_time)
print("The number of iterations is", eval[0])
print("The number of function evaluation is:", eval[1])
print("The number of gradient evaluation is:", eval[2])
#print("The mse error is:", r['error'].iloc[-1])

print()
print('-'*100)
print()

#optimal values selection 
last_param = r['param'].iloc[-1]

#prediction on train based on optimal parameters
y_hat_opt = nn_pred2(last_param, sigma, N, X_train) #omega, sigma, N, X

#final MSE on train
#e_opt= mean_squared_error(y_train,y_hat_opt)*0.5 
#print("Final MSE on Train set:",e_opt)
print("Final MSE on Train set:", r['error'].iloc[-1])
print("Regularized error on Train:", r['error'].iloc[-1] + 0.5*rho*np.linalg.norm(last_param)**2)
print()
print("Final gradient norm (Train):", np.linalg.norm(grad_E_omega2(last_param, sigma, rho, N, y_train, X_train)))
#############

#test part
#prediction on test based on optimal parameters
y_pred_test = nn_pred2(last_param, sigma, N, X_test)

e_test = mean_squared_error(y_test,y_pred_test)*0.5#empirical risk
print("MSE on Test:",e_test)
print("Regularized error on Test:", e_test + 0.5*rho*np.linalg.norm(last_param)**2)


################################################

#plot
fun_plot3(last_param, sigma, N)