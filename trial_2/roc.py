import csv
from sklearn.metrics import roc_curve,auc
import numpy as np

input_file = "20190804_roc/val_h_roc_scores25000.csv"
output_file = '20190803_roc/val_total_roc.csv'

csv_file = open(input_file,"r")
f = csv.reader(csv_file,delimiter=",")
y_t = []
y_score = []
for row in f:
    y_t.append(int(row[0]))
    y_score.append(float(row[1]))

y_t = np.array(y_t)
y_score = np.array(y_score)
fpr,tpr,a = roc_curve(y_t,y_score,pos_label=1)
print(auc(fpr,tpr))
with open(output_file,'w') as f:
    writer = csv.writer(f,lineterminator='\n')
    writer.writerow(fpr)
    writer.writerow(tpr)
    
