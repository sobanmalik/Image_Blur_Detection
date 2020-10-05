import pandas as pd
import numpy as np
import cv2
import os
from skimage.filters import laplace, sobel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import minmax_scale


#Get Paths
artifically_blurred_path = "./CERTH_ImageBlurDataset/TrainingSet/Artificially-Blurred"
naturally_blurred_path = "./CERTH_ImageBlurDataset/TrainingSet/Naturally-Blurred"
new_digi_blur_path = "./CERTH_ImageBlurDataset/TrainingSet/NewDigitalBlur"
undistorted_path = "./CERTH_ImageBlurDataset/TrainingSet/Undistorted"


# Laplace
def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


# FFT
def detect_blur_fft(image):
    # grab the dimensions of the image and use the dimensions to
    # derive the center (x, y)-coordinates
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



#PreProcessing
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

#Create DataFrames
print('\nLoading Images.......\n')
art_blur_df = create_df(artifically_blurred_path)
nat_blur_df = create_df(naturally_blurred_path)
digi_blur_df = create_df(new_digi_blur_path)
undistorted_df = create_df(undistorted_path)


#Label DataFrames
art_blur_df['label'] = 1
nat_blur_df['label'] = 1
digi_blur_df['label'] = 1
undistorted_df['label'] = -1


# Merge DataFrames
train_df = pd.concat([ nat_blur_df, undistorted_df,art_blur_df,digi_blur_df])
train_df = train_df.reset_index(drop=True)


# Normalization
train = train_df
train['Lap_var']=minmax_scale(train.Lap_var)
train['Lap_max']=minmax_scale(train.Lap_max)
train['FFT']=minmax_scale(train.FFT)
train['Sobel_var']=minmax_scale(train.Sobel_var)
train['Sobel_max']=minmax_scale(train.Sobel_max)


#Features Selection and Data Splitting 
columns = ['Lap_var', 'Lap_max','Sobel_max']#,'FFT','Sobel_var'
train_x, test_x, train_y, test_y = train_test_split( train[columns], train['label']
                                                    , test_size=0.2, random_state=42)


# Random Forest

#model=RandomForestClassifier(n_estimators=100, 
#                                  min_samples_split=10, 
#                                  min_samples_leaf=1, 
#                                  max_features=0.9,
#                                  max_depth=6, 
#                                  bootstrap=True)
# model.fit(train_x,train_y)
# predictions = model.predict(test_x)
# print('train_accuracy: ', accuracy_score(model.predict(train_x),train_y ))
# print('val_accuracy: ', accuracy_score(predictions, test_y))

# import pickle 

# filename = 'rf_model.sav'
# pickle.dump(model, open(filename, 'wb'))



# Catboost
from catboost import CatBoostClassifier
cb = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.01,
    depth=2,
#     loss_function='Logloss',
    eval_metric='Accuracy',
    metric_period = 100)  
#     leaf_estimation_method='Newton')
cb.fit(train_x, train_y,
             eval_set=(test_x,test_y),
             #cat_features=categorical_var,
             use_best_model=True,
             verbose=True)

predictions = cb.predict(test_x)
print('train_accuracy: ', accuracy_score(cb.predict(train_x),train_y ))
print('val_accuracy: ', accuracy_score(predictions, test_y))

import pickle
filename = 'model.sav'
pickle.dump(cb, open(filename, 'wb'))

