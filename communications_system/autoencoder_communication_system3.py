# -*- coding: utf-8 -*-
"""
Edited Version - Haijia

"""

# importing libs# import
import tensorflow as tf
from keras.layers import Input, Dense, GaussianNoise
from keras.models import Model
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
import random as rn



import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
'''
PLOTTING THE AUTOENCODER
'''
#plt.plot(EbNodB_range, ber, linestyle='--', color='b', marker='o', label='Autoencoder(4,7)')
import numpy as np

'''
BPSK ERROR RATE
'''
snr_array = [-4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
bpsk_2_2_array = [0.372227635, 0.3445664612, 0.3167366376, 0.2889123879, 0.261288977, 0.2340808161, 0.2075181919, 0.1818424103, 0.1572992071, 0.1341303967, 0.112563904, 0.09280255191, 0.07501225672, 0.05931057525, 0.04575681512, 0.03434508336, 0.02500163608, 0.01758762106, 0.0119077343, 0.007724463286, 0.004776581562, 0.002799609679, 0.001545349631, 0.0007975926703, 0.0003818155482]
bpsk_3_3_array = [0.5583414525, 0.5168496918, 0.4751049564, 0.4333685818, 0.3919334656, 0.3511212242, 0.3112772879, 0.2727636155, 0.2359488106, 0.201195595, 0.1688458559, 0.1392038279, 0.1125183851, 0.08896586288, 0.06863522268, 0.05151762504, 0.03750245412, 0.02638143159, 0.01786160144, 0.01158669493, 0.007164872343, 0.004199414518, 0.002318024446, 0.001196389005, 0.0005727233222]
bpsk_4_4_array = [0.7444552699, 0.6891329224, 0.6334732752, 0.5778247757, 0.5225779541, 0.4681616322, 0.4150363838, 0.3636848206, 0.3145984141, 0.2682607933, 0.2251278079, 0.1856051038, 0.1500245134, 0.1186211505, 0.09151363024, 0.06869016672, 0.05000327216, 0.03517524212, 0.02381546859, 0.01544892657, 0.009553163124, 0.005599219358, 0.003090699262, 0.001595185341, 0.0007636310963]
bpsk_5_5_array =[0.9305690874, 0.861416153, 0.791841594, 0.7222809696, 0.6532224426, 0.5852020403, 0.5187954798, 0.4546060258, 0.3932480176, 0.3353259916, 0.2814097599, 0.2320063798, 0.1875306418, 0.1482764381, 0.1143920378, 0.0858627084, 0.0625040902, 0.04396905265, 0.02976933574, 0.01931115821, 0.0119414539, 0.006999024197, 0.003863374077, 0.001993981676, 0.0009545388704]
bpsk_6_6_array =[1.116682905, 1.033699384, 0.9502099129, 0.8667371636, 0.7838669311, 0.7022424484, 0.6225545757, 0.545527231, 0.4718976212, 0.40239119, 0.3376917119, 0.2784076557, 0.2250367702, 0.1779317258, 0.1372704454, 0.1030352501, 0.07500490824, 0.05276286318, 0.03572320289, 0.02317338986, 0.01432974469, 0.008398829037, 0.004636048892, 0.002392778011, 0.001145446644]
bpsk_7_7_array =[1.302796722, 1.205982614, 1.108578232, 1.011193357, 0.9145114197, 0.8192828564, 0.7263136717, 0.6364484361, 0.5505472247, 0.4694563883, 0.3939736638, 0.3248089317, 0.2625428985, 0.2075870134, 0.1601488529, 0.1202077918, 0.08750572629, 0.06155667371, 0.04167707003, 0.0270356215, 0.01671803547, 0.009798633876, 0.005408723708, 0.002791574346, 0.001336354419]
bpsk_8_8_array =[1.48891054, 1.378265845, 1.26694655, 1.155649551, 1.045155908, 0.9363232645, 0.8300727676, 0.7273696413, 0.6291968282, 0.5365215866, 0.4502556158, 0.3712102076, 0.3000490269, 0.237242301, 0.1830272605, 0.1373803334, 0.1000065443, 0.07035048424, 0.04763093718, 0.03089785314, 0.01910632625, 0.01119843872, 0.006181398523, 0.003190370681, 0.001527262193]
bpsk_9_9_array =[1.675024357, 1.550549075, 1.425314869, 1.300105745, 1.175800397, 1.053363673, 0.9338318636, 0.8182908464, 0.7078464317, 0.603586785, 0.5065375678, 0.4176114836, 0.3375551552, 0.2668975886, 0.205905668, 0.1545528751, 0.1125073624, 0.07914429478, 0.05358480433, 0.03476008479, 0.02149461703, 0.01259824356, 0.006954073338, 0.003589167016, 0.001718169967]
bpsk_10_10_array =[1.861138175, 1.722832306, 1.583683188, 1.444561939, 1.306444885, 1.170404081, 1.03759096, 0.9092120516, 0.7864960353, 0.6706519833, 0.5628195198, 0.4640127596, 0.3750612836, 0.2965528763, 0.2287840756, 0.1717254168, 0.1250081804, 0.08793810531, 0.05953867148, 0.03862231643, 0.02388290781, 0.01399804839, 0.007726748154, 0.003987963352, 0.001909077741]
bpsk_11_11_array =[2.047251992, 1.895115537, 1.742051507, 1.589018133, 1.437089374, 1.287444489, 1.141350055, 1.000133257, 0.8651456388, 0.7377171816, 0.6191014717, 0.5104140355, 0.4125674119, 0.3262081639, 0.2516624832, 0.1888979585, 0.1375089984, 0.09673191584, 0.06549253863, 0.04248454807, 0.02627119859, 0.01539785323, 0.008499422969, 0.004386759687, 0.002099985515]
bpsk_12_12_array =[2.23336581, 2.067398767, 1.900419826, 1.733474327, 1.567733862, 1.404484897, 1.245109151, 1.091054462, 0.9437952423, 0.80478238, 0.6753834237, 0.5568153115, 0.4500735403, 0.3558634515, 0.2745408907, 0.2060705002, 0.1500098165, 0.1055257264, 0.07144640577, 0.04634677971, 0.02865948937, 0.01679765807, 0.009272097785, 0.004785556022, 0.002290893289]
bpsk_13_13_array =[2.419479627, 2.239681998, 2.058788145, 1.877930521, 1.698378351, 1.521525305, 1.348868247, 1.181975667, 1.022444846, 0.8718475783, 0.7316653757, 0.6032165874, 0.4875796687, 0.3855187391, 0.2974192983, 0.2232430418, 0.1625106345, 0.1143195369, 0.07740027292, 0.05020901136, 0.03104778015, 0.01819746291, 0.0100447726, 0.005184352357, 0.002481801063]
bpsk_14_14_array =[2.605593445, 2.411965228, 2.217156463, 2.022386715, 1.829022839, 1.638565713, 1.452627343, 1.272896872, 1.101094449, 0.9389127766, 0.7879473277, 0.6496178634, 0.525085797, 0.4151740268, 0.3202977059, 0.2404155835, 0.1750114526, 0.1231133474, 0.08335414007, 0.054071243, 0.03343607093, 0.01959726775, 0.01081744742, 0.005583148692, 0.002672708837]

auto_2_2_array = np.load('auto_2_2.npy')
auto_3_3_array = np.load('auto_3_3.npy')
auto_4_4_array = np.load('auto_4_4.npy')
auto_5_5_array = np.load('auto_5_5.npy')
auto_6_6_array = np.load('auto_6_6.npy')
auto_7_7_array = np.load('auto_7_7.npy') #0.5085225
auto_8_8_array = np.load('auto_8_8.npy') #0.577353
auto_9_9_array = np.load('auto_9_9.npy') #0.6946275
auto_10_10_array = np.load('auto_10_10.npy') #0.6925
auto_11_11_array = np.load('auto_11_11.npy') #0.7251575

plt.title('BPSK Modulation')
#plt.plot(snr_array, bpsk_2_2_array, linestyle = ':', marker = 's', color = 'lightpink', label = 'BPSK (2, 2)')
#plt.plot(snr_array, auto_2_2_array, linestyle = '-', marker = '*', color = 'dimgrey', label = 'AE (2,2)')

#plt.plot(snr_array, bpsk_3_3_array, linestyle = ':', marker = 's', color = 'lavenderblush', label = 'BPSK (3, 3)')
#plt.plot(snr_array, auto_3_3_array, linestyle = '-', marker = '*', color = 'darkgrey', label = 'AE (3,3)')

#plt.plot(snr_array, bpsk_4_4_array, linestyle = ':', marker = 's', color = 'orchid', label = 'BPSK (4, 4)')
#plt.plot(snr_array, auto_4_4_array, linestyle = '-', marker = '*', color = 'gainsboro', label = 'AE (4,4)')

#plt.plot(snr_array, bpsk_5_5_array, linestyle = ':', marker = 's', color = 'thistle', label = 'BPSK (5, 5)')
#plt.plot(snr_array, auto_5_5_array, linestyle = '-', marker = '*', color = 'brown', label = 'AE (5,5)')

#plt.plot(snr_array, bpsk_6_6_array, linestyle = ':', marker = 's', color = 'indigo', label = 'BPSK (6, 6)')
#plt.plot(snr_array, auto_6_6_array, linestyle = '-', marker = '*', color = 'r', label = 'AE (6,6)')

#plt.plot(snr_array, bpsk_7_7_array, linestyle = ':', marker = 's', color = 'b', label = 'BPSK (7, 7)')
#plt.plot(snr_array, auto_7_7_array, linestyle = '-', marker = '*', color = 'lightsalmon', label = 'AE (7,7)')

#plt.plot(snr_array, bpsk_8_8_array, linestyle = ':', marker = 's', color = 'midnightblue', label = 'BPSK (8, 8)')
#plt.plot(snr_array, auto_8_8_array, linestyle = '-', marker = '*', color = 'saddlebrown', label = 'AE (8,8)')

#plt.plot(snr_array, bpsk_9_9_array, linestyle = ':', marker = 's', color = 'cornflowerblue', label = 'BPSK (9, 9)')
#plt.plot(snr_array, auto_9_9_array, linestyle = '-', marker = '*', color = 'antiquewhite', label = 'AE (9,9)')


#plt.plot(snr_array, bpsk_10_10_array, linestyle = ':', marker = 's', color = 'steelblue', label = 'BPSK (10, 10)')
#plt.plot(snr_array, auto_10_10_array, linestyle = '-', marker = '*', color = 'palegoldenrod', label = 'AE (10,10)')

plt.plot(snr_array, bpsk_11_11_array, linestyle = ':', marker = 's', color = 'lightblue', label = 'BPSK (11, 11)')
plt.plot(snr_array, auto_11_11_array, linestyle = '-', marker = '*', color = 'yellow', label = 'AE (11,11)')

'''
plt.plot(snr_array, bpsk_12_12_array, linestyle = ':', marker = 's', color = 'cyan', label = 'BPSK (12, 12)')
plt.plot(snr_array, auto_12_12_array, linestyle = '-', marker = '*', color = 'greenyellow', label = 'AE (12,12)')

plt.plot(snr_array, bpsk_13_13_array, linestyle = ':', marker = 's', color = 'teal', label = 'BPSK (13, 13)')
plt.plot(snr_array, auto_13_13_array, linestyle = '-', marker = '*', color = 'darkseagreen', label = 'AE (13,13)')

plt.plot(snr_array, bpsk_14_14_array, linestyle = ':', marker = 's', color = 'turqoise', label = 'BPSK (14, 14)')
plt.plot(snr_array, auto_14_14_array, linestyle = '-', marker = '*', color = 'limegreen', label = 'AE (14,14)')

'''
# plt.plot(EbNodB_range, ber, linestyle='', marker='o', color='r')
# plt.plot(EbNodB_range, ber, linestyle='-', color = 'b')
# plt.plot(list(EbNodB_range), ber_theory, 'ro-',label='BPSK BER')
print(bpsk_11_11_array)
print(auto_11_11_array)

plt.yscale('log')
plt.xlabel('SNR Range')
plt.ylabel('Block Error Rate')
plt.grid()
plt.legend(loc='upper right', ncol=1)

plt.savefig('BPSK_AE_11_11')
plt.show()