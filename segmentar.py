#!/usr/bin/env python
import os
import cv2
import numpy as np
import sys
#from matplotlib import pyplot as plt

def read_files(path):
	files = []
	for r, d, f in os.walk(path):
		for file in f:
			files.append(os.path.join(r,file))

	return files			

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
	return filter(lambda x: x.size > mean, ctrs)

def find_segments(th,mask,imgIn):
	ctrs, hier = cv2.findContours(th.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	segments = []
	# sort contours
	ctrs =  mean_contours(ctrs)
	sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

	dp = map(lambda x: x.size,ctrs)
	dp = np.asarray(dp)
	dp = dp.std()

	for i, ctr in enumerate(sorted_ctrs):
		x, y, w, h = cv2.boundingRect(ctr) #ta ao contrario os valores?
		roi = mask[y:y + h, x:x + w] #aqui seria com a mascara aplicada ja
		print("x: {} y: {} w: {} h: {} size: {}".format(x,y,w,h,ctr.size))
		#achar tam imagem
		height, width, channels = imgIn.shape

		#se o contorno achado for menor que a imagem e maior que +- tam da menor semnte
		if w < width and h < height and ctr.size > dp:  # Chondrilla_juncea, Brassica_juncea, Ammi_majus don't pass some seeds 
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
			#img2 = img2.astype(np.int8)

			#segment = imgIn[y:y + h, x:x + w]
			#segment = cv2.bitwise_and(segment,segment,mask=img2)
			segments.append((img2,y,h,x,w))
	
	return segments

def main():
	files = read_files("database/") #Chondrilla_juncea, Brassica_juncea, Ammi_majus
	bad_files = []
	for f in files:
		print(f)
		imgIn = cv2.imread(f)
		th,mask = segmentate_image(imgIn)
		segments = find_segments(th,mask,imgIn)
		print(len(segments))
		cv2.imshow('aaaa',imgIn)
		cv2.waitKey()
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
