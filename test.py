# Evaluation

import pickle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import minmax_scale
import pandas as pd
import numpy as np
import cv2
import os
from skimage.filters import laplace, sobel


# FFT
def detect_blur_fft(image):
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)
    magnitude = 20 * np.log(np.abs(fftShift))
    fftShift[cY - 60:cY + 60, cX - 60:cX + 60] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)
    
    return mean


#Preprocessing
def create_df(path):
    
    import time
    tick = time.time()
    
    image_list = os.listdir(path)
    variances = []
    maxes = []
    fft_mean = []
    sobel_var = []
    sobel_max = []
    
    for img in image_list:
        image = cv2.imread(path + '/' + img)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (640,512))
        var = laplace(gray).var()
        max_ = np.amax(laplace(gray))
        mean = detect_blur_fft(gray)
        sob = sobel(gray).var()
        sob_max = np.amax(sobel(gray))
        variances.append(var)
        maxes.append(max_)
        fft_mean.append(mean)
        sobel_var.append(sob)
        sobel_max.append(sob_max)
        
    dic = {'Images': image_list, 'Lap_var': variances, 'Lap_max': maxes, 'FFT': fft_mean,
            'Sobel_var': sobel_var, 'Sobel_max': sobel_max}
    df = pd.DataFrame(dic)
    print('time_to_process_folder: ',np.round((time.time() - tick)/60,2), 'min')
    return df


#Get Paths
digi_blur_eval = "./CERTH_ImageBlurDataset/EvaluationSet/DigitalBlurSet"
naturally_blurred_eval = "./CERTH_ImageBlurDataset/EvaluationSet/NaturalBlurSet"


#Preprocess on images and create DataFrames
print('\nLoading Evaluation Images.........\n')

digi_blur_df = create_df(digi_blur_eval)
nat_blur_eval_df = create_df(naturally_blurred_eval)
#Merge
eval_df = pd.concat([digi_blur_df, nat_blur_eval_df])
eval_df

#Reading the true labels from excel
df_1 = pd.read_excel('./CERTH_ImageBlurDataset/EvaluationSet/DigitalBlurSet.xlsx')
df_1.rename(columns = {'MyDigital Blur':'Images','Unnamed: 1':'blur_label'}, inplace = True)
df_2 = pd.read_excel('./CERTH_ImageBlurDataset/EvaluationSet/NaturalBlurSet.xlsx')
df_2.rename(columns = {'Image Name': 'Images','Blur Label':'blur_label'}, inplace = True)
df_2 = df_2.sort_values('Images')
df_eval = pd.concat([df_1, df_2])

#Normalising
eval_ = eval_df
eval_['Lap_var']=minmax_scale(eval_.Lap_var)
eval_['Lap_max']=minmax_scale(eval_.Lap_max)
eval_['FFT']=minmax_scale(eval_.FFT)
eval_['Sobel_var'] = minmax_scale(eval_.Sobel_var)
eval_['Sobel_max'] = minmax_scale(eval_.Sobel_max)


#Loading Model
filename = 'model.sav'
model = pickle.load(open(filename, 'rb'))


#Predicting
columns = ['Lap_var', 'Lap_max','Sobel_max']#,'FFT','Sobel_var'
predictions = model.predict(eval_[columns])
print('Accuracy on Evaluation set is: ',accuracy_score(predictions, df_eval['blur_label']).round(2))

