
from scipy import signal
from sklearn import svm as SKSVM
from sklearn.externals import joblib

import cv2
import numpy as np

import pywt

import glob

import sys
import os

import warnings

warnings.filterwarnings("ignore")

def create_blank(width, height, rgb_color=(0, 0, 0)):

    image = np.zeros((height, width, 3), np.uint8)
    color = tuple(reversed(rgb_color))
    image[:] = color
    return image


def expandMat(img , no_of_iterations = 30) :
	imgArray = []
	count = 0
	for i in range(no_of_iterations) :

		height, width, channels = img.shape
		reduction_factor = int(( width + height ) /2 * 3.5 / 100 )
		blank_img = create_blank(width , height , (255 , 255 , 255))
		if (width - 20 > reduction_factor*i and height - 20 > reduction_factor*i) :
			img = cv2.resize(img , (width - reduction_factor*i , height - reduction_factor*i))
			blank_img[ int(reduction_factor * i / 2): int(reduction_factor * i / 2 )+ img.shape[0] , int(reduction_factor * i / 2): int(reduction_factor * i / 2 )+ img.shape[1] ] = img
			imgArray.append(blank_img)
			count = count+1
	return imgArray

svm_file_name =  os.path.join("data" , "output" , "sign.svm")
index_file_name = os.path.join("data" , "output" , "index.txt")
png_search = os.path.join("data" , "**" , "*.png")
jpg_search = os.path.join("data" , "**" , "*.jpg")
toExpand = False

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

def circle_conditions( x  , y , r) :
	if ( x *x + y*y < r*r) :
		return True
	else :
		return False

def main():

	count = 0
	png_list = glob.glob(png_search)
	jpg_list = glob.glob(jpg_search)

	img_list = png_list + jpg_list

	indexList = []
	training_set = []
	training_labels = []

	target = open( index_file_name , "w")

	for img_name in img_list :
		dir_name = img_name[img_name.find(os.path.sep) + 1 : img_name.rfind(os.path.sep)  ]
		if(dir_name == ''):
			continue
		if dir_name not in indexList :
			indexList.append(dir_name)
			target.write(dir_name + ":" + str(indexList.index(dir_name))+ "\n" )

		index = indexList.index(dir_name)
		#img = cv2.imread(img_name)
		#----------------------------------#

		image = cv2.imread(img_name)
		gray = cv2.cvtColor( image ,cv2.COLOR_RGB2GRAY )
		circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
		imgList = expandMat(image)
		errorOccoured = False
		errorOccoured_inside = False

		if circles is not None :

			circles = np.round(circles[0, :]).astype("int")
			for (x , y , r) in circles :
				errorOccoured = False
				new_img = image[ y-r:y+r , x-r:x+r]
				size = 2 * r
				
				for i in range( 0 , size -1) :
					for j in range( 0  , size -1) :
						if ( circle_conditions( i - r, j - r ,r) == False) :
							try :
								new_img[i,j] = (255 , 255 , 255)
							except :
								errorOccoured = True
								break
				if errorOccoured == True:
					continue

				if toExpand == True :

					for imgMat in imgList :
						
						gray = cv2.cvtColor( imgMat ,cv2.COLOR_RGB2GRAY )
						circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
						if circles is not None :

							circles = np.round(circles[0, :]).astype("int")
							for (x , y , r) in circles :
								errorOccoured_inside = False
								new_img = image[ y-r:y+r , x-r:x+r]
								size = 2 * r
				
								for i in range( 0 , size -1) :
									for j in range( 0  , size -1) :
										if ( circle_conditions( i - r, j - r ,r) == False) :
											try :
												new_img[i,j] = (255 , 255 , 255)
											except :
												errorOccoured_inside = True
												break
								if errorOccoured_inside == True:
									continue
								img = waveletTransform([new_img])
								img = cv2.resize(img , (100 , 100))
								flat_arr = img.flatten()

								training_set.append( flat_arr )
								training_labels.append( index )

								count = count+1
								

				
				img = waveletTransform(new_img)
				img = cv2.resize(img , (100 , 100))
				flat_arr = img.flatten()

				training_set.append( flat_arr )
				training_labels.append( index )

				count = count + 1
				#print(count)

		

	trainData = np.array(training_set)
	responses = np.array(training_labels)

	support_vector = SKSVM.SVC()
	support_vector.fit(trainData, responses)

	joblib.dump(support_vector , svm_file_name , compress = 1)

	#print(str(count))



if __name__ == '__main__':
	if len(sys.argv) >= 2 :
		if sys.argv[1] == "--expand" :
			toExpand = True
	main()
