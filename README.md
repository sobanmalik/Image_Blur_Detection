# Image_Blur_Detection

Classification of Blurred and Non-Blurred Images (CERTH Image Blur Dataset)

Approach Used: Laplacian, FFT and Sobel Operators are applied on each of the input images, which are resclaed to a shape of (640,512). 	Variances and Maxes of Laplacian and Sobel Operators, and mean of FFT are used to create features. The features are trained using tree based machine learning techniques. The best accuracy of 88.5% was obtained on evaluation set using Catboost Classifier. 


Download Data: Download the CERTH Image Blur Dataset and put it in the folder named 'CERTH_ImageBlurDataset', in the main directory.

Training: Preprocess all the input training images from Dataset and apply above mentioned filters to them. Train a tree based model on the features.
Run the 'train.py' to achieve the same. The model will be saved as 'model.sav' in the main directory.

Evaluation: Use the model to predict on the evaluation set. Preprocess the images and load the model 'model.sav'. Predict on the images using this model.
Run the 'test.py' to achieve the same. It will output the accuracy on the evaluation set.

Inferencing: To inference your own images, load them into the folder name 'Test_own_images' and run the 'Inference.py' script.

Packages to be installed: OpenCV, Scikit-learn, Scikit-image, Pandas, Numpy, Catboost, Pickle
