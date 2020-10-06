# This scripts kept a record of F1 score over adasyn percentage for data set racketsports imbalance ratio 11:108

import matplotlib.pyplot as plt

ada_perc = [0,0.1,0.3,0.5,0.7,0.9,1]
f1_mean_max = [(0.28832252199125263, 0.47619047619047616),
 (0.36971497370878176, 0.47619047619047616),
 (0.4225913577306766, 0.5263157894736842),
 (0.4085727799891887, 0.47619047619047616),
 (0.31591589267285863, 0.5263157894736842),
 (0.151973951973952, 0.16666666666666669),
(0.17521367521367523, 0.3076923076923077)]
f1_mean = [i[0] for i in f1_mean_max]
f1_max = [i[1] for i in f1_mean_max]

plt.plot(ada_perc,f1_mean,'b*',label='average_F1_score (from 10 runs)')
plt.plot(ada_perc,f1_max, 'g*',label='max_F1_score (from 10 runs)')
plt.ylim(top=0.6,bottom=0)
plt.ylabel('F1_score')
plt.xlabel('percentage of Adasyn samples')
plt.legend()
plt.savefig('/Users/jiedali/Documents/research/notes/plots/' + 'f1_vs_ada_perc_RacketSport.png')

