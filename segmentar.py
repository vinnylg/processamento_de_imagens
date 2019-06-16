#!/usr/bin/env python
import os
import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt

def read_files(path):
	files = []
	for r, d, f in os.walk(path):
		for file in f:
			files.append(os.path.join(r,file))

	return files	

def filter_fourier(img):
	f = np.fft.fft2(img)
	fshift = np.fft.fftshift(img)
	magnitude_spectrum = 20*np.log(np.abs(fshift))
	cv2.imshow('magnitude_spectrum',magnitude_spectrum)
	cv2.waitKey()
	
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
		if ctr.size >= dp and w < width and h < height:  # Chondrilla_juncea, Brassica_juncea, Ammi_majus don't pass some seeds 
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

def main():
	files = read_files("database") #Chondrilla_juncea, Brassica_juncea, Ammi_majus
	bad_files = []
	for f in files:
		print(f)
		imgIn = cv2.imread(f)
		cv2.imshow(f,imgIn)
		cv2.waitKey()
		filter_fourier(imgIn)
		th,mask = segmentate_image(imgIn)
		segments = find_segments(th,mask,imgIn)
		print(len(segments))
		for seg in segments:
			im,y,h,x,w = seg
			# print("y:{} h:{} x:{} w:{}".format(y,h,x,w))
			# cv2.imshow('seg th',im)
			# cv2.waitKey()

			imcor = imgIn[y:y + h, x:x + w]
			im = im.astype(np.int8)
			imcor = cv2.bitwise_and(imcor,imcor,mask=im)
			cv2.imshow('seg ori',imcor)
			cv2.waitKey()
		
		print("")
	for bad in bad_files:
		print bad

if __name__== "__main__":
  main()
#
