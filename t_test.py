# paired t-test to compare EM vs ADASYN, EM vs SMOTE
from scipy import stats
em=[0.361,0.197,0.689,0.843]
ada=[0.278,0.164,0.462,0.762]
smote=[0.194,0.167,0.490,0.864]
stats.ttest_rel(em,ada)
stats.ttest_rel(em,smote)