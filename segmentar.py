#!/usr/bin/env python
import os
import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt
import pandas as pd

def read_files(path):
	files = []
	for r, d, f in os.walk(path):
		for file in f:
			files.append(os.path.join(r,file))

	return files	

def sharping(img):
	kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
	return cv2.filter2D(img,-1,kernel)
	
def segmentate_image(imgIn):
	hsv = cv2.cvtColor(imgIn, cv2.COLOR_BGR2HSV)

	lower_blue = 100, 150, 0
	upper_blue = 140, 255, 255

	th = cv2.inRange(hsv, lower_blue, upper_blue)

	mask = 255 - th
	return th,mask
 
def find_first_big_var(arr):
	arr = sorted(arr)
	prev = var = nxt = first_big_var = 0
	for i in range(len(arr)):
		nxt = 0
		var = np.abs(arr[i]-prev)

		if(i+1 != len(arr)):
			nxt = np.abs(arr[i+1]-arr[i])

		if var > prev and var > nxt:
			big_var = var
		
		prev = arr[i]
	return big_var

def metrics_contours(ctrs):
	sizes = map(lambda x: str(x.size), ctrs)
	sizes = list(dict.fromkeys(sizes))
	sizes = map(int,sizes)
	var = find_first_big_var(sizes)

	mean = sum(sizes)/len(sizes)
	ctrs = filter(lambda x: x.size > mean, ctrs)
	
	dp = map(lambda x: x.size,ctrs)
	dp = np.asarray(dp)
	dp = dp.std()

	if (mean < dp):
		sizes = map(lambda x: x.size, ctrs)
		mean = sum(sizes)/len(sizes)

		ctrs = filter(lambda x: x.size > mean, ctrs)
		dp = map(lambda x: x.size,ctrs)
		dp = np.asarray(dp)
		dp = dp.std()


	return ctrs, mean, dp, var

def find_segments(th,mask,imgIn):
	ctrs, hier = cv2.findContours(th.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	segments = []
	# sort contours
	sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
	ctrs, mean, dp, var =  metrics_contours(ctrs)
	faixa = mean/2 if mean>dp else dp/2
	faixa = np.around(faixa,0)
	print("mean: {} dp: {} faixa: {} big_var: {}".format(mean,dp,faixa,var))
	for i, ctr in enumerate(sorted_ctrs):
		x, y, w, h = cv2.boundingRect(ctr) #ta ao contrario os valores?
		roi = mask[y:y + h, x:x + w] #aqui seria com a mascara aplicada ja
		#achar tam imagem
		height, width, channels = imgIn.shape
		#se o contorno achado for menor que a imagem e maior que +- tam da menor semnte
		# if ctr.size > 400:
		#  	print("{}:{}".format(ctr.size,ctr.size > faixa or ctr.size >= var))
		if (ctr.size >= faixa or ctr.size >= var) and w < width and h < height:  # Chondrilla_juncea, Brassica_juncea, Ammi_majus don't pass some seeds 
			image = roi.astype('uint8')
			nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
			sizes = stats[:, -1]
			max_label = 1
			max_size = sizes[1]
			for i in range(2, nb_components):
				if sizes[i] > max_size:
					max_label = i
					max_size = sizes[i]
			img2 = np.zeros(output.shape)
			img2[output == max_label] = 255
			segments.append((img2,y,h,x,w))

	return segments

def calcHuMoments(img):
	img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret,th = cv2.threshold(img,50,255,cv2.THRESH_BINARY_INV)
	moment = cv2.moments(img)
	huMoment = cv2.HuMoments(moment)
	return map(lambda hu: -1 * np.sign(hu) * np.log10(np.abs(hu)), huMoment)
 

def main():
	files = read_files("database") #Chondrilla_juncea, Brassica_juncea, Ammi_majus
	files = sorted(files)
	huMoments = []
	i = 0
	for f in files:
		print('segmenting {} ...'.format(f))
		imgIn = cv2.imread(f)
		th,mask = segmentate_image(imgIn)
		segments = find_segments(th,mask,imgIn)
		print('segments: {}'.format(len(segments)))
		imgin = cv2.resize(imgIn,None,fx=0.5,fy=0.5)
		# cv2.imshow(f,imgin)
		# cv2.moveWindow(f,0,0)
		# cv2.waitKey()
		for seg in segments:
			i = i+1
			im,y,h,x,w = seg
			im = im.astype(np.int8)

			imcor = imgIn[y:y + h, x:x + w]
			imcor = cv2.bitwise_and(imcor,imcor,mask=im)

			cv2.imwrite('output/{}.jpg'.format(i),imcor)
			imcor = sharping(imcor)

			huMoment = calcHuMoments(imcor)
			huMoment = np.concatenate(huMoment, axis=0)
			huMoment = huMoment.tolist()
			huMoments.append(huMoment)

		cv2.destroyAllWindows()

	huMoments = np.array(huMoments)

	df = pd.DataFrame({	'HU1':huMoments[:,0],
						'HU2':huMoments[:,1],
						'HU3':huMoments[:,2],
						'HU4':huMoments[:,3],
						'HU5':huMoments[:,4],
						'HU6':huMoments[:,5],
						'HU7':huMoments[:,6]})

	df.to_csv("huMoments.csv", encoding='utf-8',index = False)

if __name__== "__main__":
  main()
#
