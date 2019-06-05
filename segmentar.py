#!/usr/bin/env python
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

def read_files(path):
	files = []
	for r, d, f in os.walk(path):
		for file in f:
			files.append(os.path.join(r,file))

	return files			


def main():

	files = read_files("database")

	for f in files:
		imgIn = cv2.imread(f)
		hsv = cv2.cvtColor(imgIn, cv2.COLOR_BGR2HSV)

		lower_blue = 100, 150, 0
		upper_blue = 140, 255, 255

		th = cv2.inRange(hsv, lower_blue, upper_blue)

		cv2.imshow('a',th)

		mask = 255 - th

		ctrs, hier = cv2.findContours(th.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

		# sort contours
		sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

		for i, ctr in enumerate(sorted_ctrs):
			x, y, w, h = cv2.boundingRect(ctr)
			roi = th[y:y + h, x:x + w]

			if( w > 20 ):
				ctrs2, hier2 = cv2.findContours(roi.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
				sorted_ctrs2 = sorted(ctrs2, key=lambda ctr2: cv2.boundingRect(ctr2)[0])		

				for i2, ctr2 in enumerate(sorted_ctrs2):
					x2, y2, w2, h2 = cv2.boundingRect(ctr2) 
					roi2 = th[y2:y2+h2, x2:x2+w2] 

					if w2 < 20 and h2 < 20:
						roi[y2:y2+h2, x2:x2+w2] = 0
			
					mask = 255 - roi
					roi_m = cv2.bitwise_and(roi,roi, mask= mask)
					print(f.strip('.jpg')+str(i)+'.jpg')
					nome = f.split('/')
					nome = nome[2]
					cv2.imwrite(nome+'-'+str(i)+'.jpg',roi_m)

if __name__== "__main__":
  main()
#
