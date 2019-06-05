import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def read_files(path):
	files = []
	for r, d, f in os.walk(path):
		for file in f:
			files.append(os.path.join(r,file))

	return files			





def main():

	print("aaa")

	files = read_files("database")

	for f in files:
		imgIn = cv.imread(f)
		hsv = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)

		lower_blue = 100, 150, 0
		upper_blue = 140, 255, 255

		th = cv2.inRange(hsv, lower_blue, upper_blue)

		cv2.imshow('th', th)
		cv2.waitKey()

		mask = 255 - th

		cv2.imshow('mask',mask)
		cv2.waitKey()

		img_m = cv2.bitwise_and(img_in,img_in, mask= mask)

		cv2.imshow('img_m',img_m)
		cv2.waitKey()

		ctrs, hier = cv2.findContours(th.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

		# sort contours
		sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

		for i, ctr in enumerate(sorted_ctrs):
		    # Get bounding box
		    x, y, w, h = cv2.boundingRect(ctr)

		    # Getting ROI
		    roi = img_m[y:y + h, x:x + w]

		    # show ROI
		    # cv2.imshow('segment no:'+str(i),roi)
		    # cv2.rectangle(img_m, (x, y), (x + w, y + h), (0, 255, 0), 2)

if __name__== "__main__":
  main()
#
