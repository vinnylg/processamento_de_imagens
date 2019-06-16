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
 
def mean_contours(ctrs):
	mean = 0
	sizes = map(lambda x: x.size, ctrs)
	mean = sum(sizes)/len(sizes)
	return filter(lambda x: x.size > mean, ctrs), mean

def find_segments(th,mask,imgIn):
	ctrs, hier = cv2.findContours(th.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	segments = []
	# sort contours
	ctrs, mean =  mean_contours(ctrs)
	sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

	dp = map(lambda x: x.size,ctrs)
	dp = np.asarray(dp)
	dp = dp.std()

	sizes = map(lambda x: x.size, ctrs)
	smin = min(sizes)
	smax = max(sizes)
	for i, ctr in enumerate(sorted_ctrs):
		x, y, w, h = cv2.boundingRect(ctr) #ta ao contrario os valores?
		roi = mask[y:y + h, x:x + w] #aqui seria com a mascara aplicada ja
		# print("x: {} y: {} w: {} h: {} size: {} dp: {}".format(x,y,w,h,ctr.size,dp))
		# print("min+dp: {}\tmean: {}\tmax-dp: {}".format(smin+dp,mean,smax-dp))
		#achar tam imagem
		height, width, channels = imgIn.shape
		# print("(ctr.size)>=dp: {} ctr.size: {} dp: {}".format((ctr.size)>=dp,ctr.size,dp))
		#se o contorno achado for menor que a imagem e maior que +- tam da menor semnte
		if ctr.size >= dp-smin and w < width and h < height:  # Chondrilla_juncea, Brassica_juncea, Ammi_majus don't pass some seeds 
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
	for f in files:
		print('segmenting {} ...'.format(f))
		imgIn = cv2.imread(f)
		th,mask = segmentate_image(imgIn)
		segments = find_segments(th,mask,imgIn)
		print('segments: {}'.format(len(segments)))
		imgin = cv2.resize(imgIn,None,fx=0.75,fy=0.75)
		cv2.imshow(f,imgin)
		cv2.moveWindow(f,0,0)
		cv2.waitKey()

		for seg in segments:
			im,y,h,x,w = seg
			im = im.astype(np.int8)

			imcor = imgIn[y:y + h, x:x + w]
			imcor = cv2.bitwise_and(imcor,imcor,mask=im)

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
