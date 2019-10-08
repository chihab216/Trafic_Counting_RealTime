
import pandas as pd
import numpy as np
import cv2


data = pd.read_csv('ground truth.csv')
preds = pd.read_csv('preds.csv')


data['xmin_pred'] = preds.xmin
data['ymin_pred'] = preds.ymin
data['xmax_pred'] = preds.xmax
data['ymax_pred'] = preds.ymax



def IOU(df):
    # determining the minimum and maximum -coordinates of the intersection rectangle
    xmin_inter = max(df.xmin, df.xmin_pred)
    ymin_inter = max(df.ymin, df.ymin_pred)
    xmax_inter = min(df.xmax, df.xmax_pred)
    ymax_inter = min(df.ymax, df.ymax_pred)
 
    # calculate area of intersection rectangle
    inter_area = max(0, xmax_inter - xmin_inter + 1) * max(0, ymax_inter - ymin_inter + 1)
 
    # calculate area of actual and predicted boxes
    actual_area = (df.xmax - df.xmin + 1) * (df.ymax - df.ymin + 1)
    pred_area = (df.xmax_pred - df.xmin_pred + 1) * (df.ymax_pred - df.ymin_pred+ 1)
 
    # computing intersection over union
    iou = inter_area / float(actual_area + pred_area - inter_area)
 
    # return the intersection over union value
    return iou


# creating a evaluation table
eval_table = pd.DataFrame()
eval_table['image_name'] = data.image_name

#applying IOU function over each image in the dataframe
eval_table['IOU'] = data.apply(IOU, axis = 1)

# creating column 'TP/FP' which will store TP for True positive and FP for False positive
# if IOU is greater than 0.5 then TP else FP
eval_table['TP/FP'] = eval_table['IOU'].apply(lambda x: 'TP' if x>=0.5 else 'FP')


# calculating Precision and recall

Precision = []
Recall = []

TP = FP = 0
FN = len(eval_table['TP/FP']== 'TP')
for index , row in eval_table.iterrows():     
    
    if row.IOU > 0.5:
        TP =TP+1
    else:
        FP =FP+1    
    

    try:
        
        AP = TP/(TP+FP)
        Rec = TP/(TP+FN)
    except ZeroDivisionError:
        
        AP = Recall = 0.0
    
    Precision.append(AP)
    Recall.append(Rec)


eval_table['Precision'] = Precision
eval_table['Recall'] = Recall



#calculating Interpolated Precision
eval_table['IP'] = eval_table.groupby('Recall')['Precision'].transform('max')




prec_at_rec = []

for recall_level in np.linspace(0.0, 1.0, 11):
    try:
        x = eval_table[eval_table['Recall'] >= recall_level]['Precision']
        prec = max(x)
    except:
        prec = 0.0
    prec_at_rec.append(prec)
avg_prec = np.mean(prec_at_rec)
print('11 point precision is ', prec_at_rec)
print('\nmap is ', avg_prec)






