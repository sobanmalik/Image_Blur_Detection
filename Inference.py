import time
import pickle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import minmax_scale
import pandas as pd
import numpy as np
import cv2
import os
from skimage.filters import laplace, sobel


tic = time.time()

#FFT
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
    #print('time_to_load_images: ',np.round((time.time() - tick)/60,2), 'min')
    return df

#Load Images
img_path = "./Test_own_images"


#Preprocess images and create DataFrame

print('\nLoading Images....\n')

img_df = create_df(img_path)
eval_df = img_df
eval_df['Lap_var']=minmax_scale(eval_df.Lap_var)
eval_df['Lap_max']=minmax_scale(eval_df.Lap_max)
eval_df['FFT']=minmax_scale(eval_df.FFT)
eval_df['Sobel_var'] = minmax_scale(eval_df.Sobel_var)
eval_df['Sobel_max'] = minmax_scale(eval_df.Sobel_max)


#Load Model
filename = 'model.sav'
model = pickle.load(open(filename, 'rb'))

#Predict
columns = ['Lap_var', 'Lap_max','Sobel_max']#,'FFT','Sobel_var'
predictions = model.predict(eval_df[columns])


for i in range(len(img_df.Images)):
    if predictions[i] == -1:
        print(str(img_df.Images[i]), ' is Not Blurry\n')
    else:
        print(str(img_df.Images[i]), ' is Blurry\n')

print('Total time to predict on images: ',np.round((time.time() - tic),2), 'seconds')

