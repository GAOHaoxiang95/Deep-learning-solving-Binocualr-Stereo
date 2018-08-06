#author: Haoxiang Gao 25/03/2018
#This flle is for reading images or image patches

import os
import caffe
import numpy as np
import cv2
def readIm(row, col, im, im2, d):
#combine two 3-channel image patches into a single 6-channel patch
	elist = [[0.00]*243]	
	newimage = np.array(elist)
	newimage.resize(9,9,3)
	
	nlist = [[0.00]*243]
	goalPatch = np.array(nlist)
	goalPatch.resize(9,9,3)

	for i in range(row - 4, row + 5):
		for j in range(col -4, col + 5):
			for k in range(0,3):
				newimage[i-row + 4,j-col + 4,k] = im[i,j,k]
				goalPatch[i-row + 4,j-col + 4,k] = im2[i,j - d,k]
				
				
	tlist = [[0.00]*486]
	test = np.array(tlist)
	test.resize(9,9,6)
	
	#combine together
	for i in range(0,9):
		for j in range(0,9):
			for k in range(0,6):
				if k < 3:
					test[i,j,k] = newimage[i,j,k]
				else:
					test[i,j,k] = goalPatch[i,j,k-3]
				
	return test

def extractPatch(row, col,img, coef):
#extract patches for displaying or comparing
	elist = [[0.0]*81*coef*coef*3]
	newimage = np.array(elist)
	newimage.resize(9*coef,9*coef,3)
	
	im = cv2.imread(img)

	if coef == 1:
		for i in range(row - 4, row + 5):
			for j in range(col -4, col + 5):
				newimage[i-row+4 ,j-col+4,0] = im[i,j,0]/255.0
				newimage[i-row+4 ,j-col+4,1] = im[i,j,1]/255.0
				newimage[i-row+4 ,j-col+4,2] = im[i,j,2]/255.0
	
	else:#generate patches to be displayed
		for i in range(row - 4, row + 5):
			for j in range(col -4, col + 5):
				for enlargei in range((i-row+4)*coef, (i-row+5)*coef):
					for enlargej in range((j-col+4)*coef, (j-col+5)*coef):
						for k in range(0,3):
							newimage[enlargei ,enlargej,k] = int(im[i,j,k])
	return newimage

def extractAllPatches(startRow, startCol, numberOfRow, numberOfCol, img):#extract patches for all pixels
	list = [([0]*(numberOfCol+startCol+2)) for i in range(startRow+numberOfRow+2)]
	for i in range(startRow, startRow + numberOfRow):
		for j in range(startCol, startCol + numberOfCol):
			list[i][j] = extractPatch(i, j, img, 1)
		print i
	return list
			