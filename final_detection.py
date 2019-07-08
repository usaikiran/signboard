
from scipy import signal
from sklearn import svm as SKSVM
from sklearn.externals import joblib

import cv2
import numpy as np
import os
import sys

import glob
import glob2

import pywt

import warnings
import time

warnings.filterwarnings("ignore")

def circle_conditions( x  , y , r) :
	if ( x *x + y*y <= r*r) :
		return True
	else :
		return False


def waveletTransform(img, mode='haar', level=1):

    imArray = 		img
    imArray = 		cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    imArray =  		np.float32(imArray)   
    imArray /= 		255;

    coeffs = 		pywt.wavedec2(imArray, mode, level=level)
    coeffs_H = 		list(coeffs)  
    coeffs_H[0] *= 	0;  

    imArray_H = 	pywt.waverec2(coeffs_H, mode);
    imArray_H *= 	255;
    imArray_H =  	np.uint8(imArray_H)

    return imArray_H


def get_prediction( image ) :

	data_path  =  os.path.join("data" , "output" , "sign.svm")
	img = waveletTransform(image)
	img = cv2.resize(img , (100 , 100))
	flat_arr = img.flatten()
	support_vector = joblib.load(data_path)
	predicted_output = support_vector.predict(flat_arr)
	return predicted_output

def main() :

	if len(sys.argv) >= 2 :
		image_name = sys.argv[1]
	else :
		print("Usage : python final_detection.py <IMAGE_PATH>")
		sys.exit()
	image = cv2.imread(image_name)
	gray = cv2.cvtColor( image ,cv2.COLOR_RGB2GRAY )
	circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)

	if circles is not None :
		circles = np.round(circles[0, :]).astype("int")
		for (x , y , r) in circles :

			new_img = image[ y-r:y+r , x-r:x+r]
			size = 2 * (r)
			for i in range( 0 , size -1) :
				for j in range( 0  , size -1) :
					if ( circle_conditions( i - r, j - r ,r) == False) :
						try:
							new_img[i,j] = (255 , 255 , 255)
						except :
							print("The input image is not compatible")
			predicted_index = get_prediction(new_img)
			print(int(predicted_index))

	else :
		print ("No signs detected")




if __name__ == '__main__':
	start = time.time()
	main()
	print(time.time() - start)


