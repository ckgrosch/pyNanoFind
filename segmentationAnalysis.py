import numpy as np
import skimage.io as skio
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def intensityMetrics(images):
    """Get lists of min, max, and average intensity value across images"""
    maxes = []
    mins = []
    averages = []
    for img in images:
        maxes.append(img.max())
        mins.append(img.min())
        averages.append(img.mean())
    maxes = np.asanyarray(maxes)
    mins = np.asanyarray(mins)
    averages = np.asanyarray(averages)
    return maxes, mins, averages

def create_metrics_dataframe(exprmnt_labels,zol_list,bas_list,prcsn_list,rcll_list,f1_list):
    """Create a Pandas dataframe containing all the metrics for each
    segmentation map in a stack of predicted segmentation maps, their associated
     experiment and index in the stack"""
    tosave = input('Do you wish to save the created pandas dataframe - y or n?')
    if tosave == 'y':
        filename = input('set name for saved dataframe')
        dfdir = input('Paste directory path where dataframe file should go')
    d = {'experiment label':exprmnt_labels,'zero one loss':zol_list,'balanced accuracy score': bas_list,'precision':prcsn_list,'recall':rcll_list,'F1 score':f1_list}
    df = pd.DataFrame(data=d)
    expmnt_df =expmntListDF(exprmnt_labels)
    df = pd.concat([df,expmnt_df],axis = 1)
    if tosave == 'y':
        df.to_csv(dfdir+filename+'.csv')
    return df

def experimentLabelsList():
    """Takes txt file and outputs list from txt file"""
    filename = input('Input directory and filename of experiment label txt file')
    file = open(filename,'r')
    expmntlist = [line.split('\n')[0] for line in file.readlines()]
    return expmntlist

def expmntListDF(expmnt_list):
    """Makes a pandas dataframe from the information in the experiment list obtained from the experiment list text file
    made using experimentLabelsList function"""
    keys = []
    mags = []
    imgnums = []
    imgparts = []
    explabels = []
    for expmnt in expmnt_list:
        parts = expmnt.split('_')
        while '' in parts:
            parts.remove('')
        if len(parts) == 6:
            explabel = parts[0]+'_'+parts[1]
            mag = parts[3]
            imgnum = parts[4]
            imgpart = parts[5]
            key = parts[0]+'_'+parts[1]+'_'+str(imgnum)
        elif len(parts) == 8:
            explabel = parts[0]+'_'+parts[2]
            key = parts[0]+'_'+parts[2]+'_'+str(parts[6])
            mag = parts[4]
            imgnum = parts[6]
            imgpart = parts[7]
        keys.append(key)
        mags.append(mag)
        imgnums.append(imgnum)
        imgparts.append(imgpart)
        explabels.append(explabel)
    df = pd.DataFrame(data = {'experiment':explabels,'keys':keys, 'mags':mags,'imgnum':imgnums, 'imgparts':imgparts})
    return df

def binarize_map(y_predOG,threshold):
    y_pred = y_predOG.copy()
    if y_pred.flags.owndata != True:
        raise RuntimeError('Data copy did not work - stopped before data overwrite occurred')
    for y in y_pred:
        y[y > threshold] = 1
        y[y <= threshold] = 0
    return y_pred

def intensityMetrics(images):
    """Get lists of min, max, and average intensity value across images"""
    maxes = []
    mins = []
    averages = []
    for img in images:
        maxes.append(img.max())
        mins.append(img.min())
        averages.append(img.mean())
    return maxes, mins, averages

def tabulate_metrics(y_pred,y_true):
    """Returns five lists containing the zero_one_loss, balanced_accuracy_score
    precision, recall, and f1 score for each segmentation map that has been
    input. Inputs are the predicted maps and true maps."""
    zol_list = []
    bas_list = []
    prcsn_list = []
    rcll_list = []
    f1_list = []
    for idx, seg in enumerate(y_true):
        zol = metrics.zero_one_loss(seg.flatten(),y_pred[idx,:,:,:].flatten()) #fix INDEXING KATE NEED TO DETERMINE INPUT shape
        zol_list.append(zol)
        bas = metrics.accuracy_score(seg.flatten(),y_pred[idx,:,:,:].flatten())
        bas_list.append(bas)
        precision, recall, f1, _ = metrics.precision_recall_fscore_support(seg.flatten(),y_pred[idx,:,:,:].flatten())
        prcsn_list.append(precision)
        rcll_list.append(recall)
        f1_list.append(f1)
    return zol_list,bas_list,prcsn_list,rcll_list,f1_list

def loadDefocusFiles(fileDir):
    """Takes in a directory which contains csv files containing the defocus and other experimental parameters"""
    files = glob(fileDir+'/*.csv')
    for idx, file in enumerate(files):
        label = file.split('-')[-1].split('.')[0]
        defocus = pd.read_csv(file)
        defocus.insert(0,'Experiment Label',label)
        if idx == 0:
            df = defocus.copy()
        else:
            df = pd.concat([df,defocus],ignore_index=True)
    df = df.rename(index = str, columns={'Unnamed: 3':'Notes'})
    df = df.drop(columns = 'Unnamed: 4')
    df['Img#'] = df['Img#'].astype(dtype='str',copy=False)
    newImgnum = []
    for data in df['Img#']:
        if len(data) == 1:
            data = '000'+ data
        elif len(data) == 2:
            data = '00' + data
        elif len(data) == 3:
            data = '0' + data
        newImgnum.append(data)
    df['Img#'] = newImgnum
    df['keys'] = df['Experiment Label'] + '_' + df['Img#']
    return df

def containsParticles(maps):
    """Function takes in segmentation maps in Keras tensor format and returns a list containing whether
    each map contained a particle written as a boolean"""
    particlelist = []
    count = 0
    for segmap in maps:
        if 1.0 in segmap:
            particlelist.append(True)
            count += 1
        else:
            particlelist.append(False)
    print('Fraction of maps which contained particles: {0}'.format(count/maps.shape[0]))
    return particlelist

def finalDataframe(predictions,imgtensor,maptensor,defocusDF, threshold = 0.6):
    """Create final data frame for a given set of input images.
    Inputs are the predictions made by the ml model, the input images in a keras tensor,
    the answer key maps tensor, and the defocus dataframe.
    Threshold sets the probability which must be achieved for the particle class"""
    expmntlist = experimentLabelsList()
    metricDF = expmntListDF(expmntlist)
    predictions = binarize_map(predictions,threshold)
    metrics = tabulate_metrics(predictions,maptensor)
    intMetrics = intensityMetrics(imgtensor)
    particlelist = containsParticles(maptensor)
    metricDF['zero-one-loss'] = metrics[0]
    metricDF['accuracy'] = metrics[1]
    metricDF['precision'] = metrics[2]
    metricDF['recall'] = metrics[3]
    metricDF['f1'] = metrics[4]
    metricDF['max'] = intMetrics[0]
    metricDF['min'] = intMetrics[1]
    metricDF['average']= intMetrics[2]
    metricDF['contains particles']= particlelist
    DF = pd.merge(defocusDF,metricDF,on='keys',how='inner',sort=True,validate='1:m')
    return DF

def find_thresh(maps,predicted,thresh = 0.7,step = 0.05,layer = 1,cutoff=0.0001):
    predBin = binarize_map(predicted[:,:,:,layer],thresh)
    acc = metrics.accuracy_score(maps[:,:,:,layer].flatten(),predBin.flatten())
    dif = 1
    loops = 30
    predBin = binarize_map(predicted[:,:,:,layer],thresh+step)
    up = metrics.accuracy_score(maps[:,:,:,layer].flatten(),predBin.flatten())
    predBin = binarize_map(predicted[:,:,:,layer],thresh-step)
    down = metrics.accuracy_score(maps[:,:,:,layer].flatten(),predBin.flatten())
    if round(up*1000) >= round(acc*1000):
        thresh += step
        print('threshold going up')
        while dif >= cutoff and loops > 0:
            if loops != 30:
                thresh += step
            if thresh >= 1:
                break
            predBin = binarize_map(predicted[:,:,:,layer],thresh)
            acc2 = metrics.accuracy_score(maps[:,:,:,layer].flatten(),predBin.flatten())
            dif = acc2 - acc
            loops -= 1
            if dif >= cutoff:
                acc = acc2
            print(acc2,thresh)
        thresh -= step
    elif round(down*1000) >= round(acc*1000):
        thresh -= step
        print('threshold going down')
        while dif > cutoff and loops > 0:
            if loops != 30:
                thresh -= step
            if thresh >= 1:
                print('break')
                break
            predBin = binarize_map(predicted[:,:,:,layer],thresh)
            acc2 = metrics.accuracy_score(maps[:,:,:,layer].flatten(),predBin.flatten())
            dif = acc2 - acc
            loops -= 1
            if dif > cutoff:
                acc = acc2
            print(acc2,thresh)
        thresh += step
    else:
        pass
    return(acc,thresh)

def plot_confusion_matrix(Y, Y_pred, Y_labels,save=False):
    """Creates a confusion matrix for the different classes given a true labels,predictions, the dataset
    and the desired trained classfier"""
    cfm = metrics.confusion_matrix(Y, Y_pred)
    cfm = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis]
    df_cfm = pd.DataFrame(data = cfm, columns=Y_labels, index=Y_labels)
    plt.subplots(figsize=(5,5))
    ax = sns.heatmap(df_cfm, annot=True,cmap='rainbow')
    ax.set(ylabel='True label',xlabel='Predicted label')
    if save == True:
        fname = input('Specify filename to save figure to: ')
        fig = ax.get_figure()
        fig.savefig(fname)
