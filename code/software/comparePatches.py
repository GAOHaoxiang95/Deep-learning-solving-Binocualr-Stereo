#author: Haoxiang Gao 25/03/2018
#This flle is for compareing patches and match patches in different methods
import caffe
import numpy as np
import readImage as r

import cv2
import copy
import math
def comparePatches(path1, path2, row1,col1,row2,col2,deploy, caffe_model,mean_f):
	
	mean_file='MEAN_FILE.npy'
	
	im = cv2.imread(path1)/255.0
	im2 = cv2.imread(path2)/255.0
	mean_npy = np.load(mean_file)
	mean = mean_npy.mean(1).mean(1)
	
	if mean_f == 1:#use mean file or not
		net = caffe.Classifier(deploy, caffe_model, mean = mean, raw_scale = 255,channel_swap=(2,1,0,5,4,3))
	else:
		net = caffe.Classifier(deploy, caffe_model, raw_scale = 255,channel_swap=(2,1,0,5,4,3))
	net.blobs['data'].reshape(1,6,9,9)
	
	elist = [[0.00]*243]
	
	newimage = np.array(elist)
	newimage.resize(9,9,3)
	
	nlist = [[0.00]*243]
	goalPatch = np.array(nlist)
	goalPatch.resize(9,9,3)
	
	for i in range(row1 - 4, row1 + 5):
		for j in range(col1 -4, col1 + 5):
			for k in range(0,3):
				newimage[i-row1 + 4,j-col1 + 4,k] = im[i,j,k]
	
	for i in range(row2 - 4, row2 + 5):
		for j in range(col2 -4, col2 + 5):
			for k in range(0,3):
				goalPatch[i-row2 + 4,j-col2 + 4,k] = im2[i,j,k]
				
	tlist = [[0.0]*486]
	test = np.array(tlist)
	test.resize(9,9,6)
	
	for i in range(0,9):
		for j in range(0,9):
			for k in range(0,6):
				if k < 3:
					test[i,j,k] = newimage[i,j,k]
				else:
					test[i,j,k] = goalPatch[i,j,k-3]
		
	output = net.predict([test])
	predictions = output[0]		
	matchingCost = predictions[0]
	return matchingCost

def comparePatches_ED(path1, path2, row1,col1,row2,col2):
	
	im = cv2.imread(path1)/255.0
	im2 = cv2.imread(path2)/255.0
	
	
	elist = [[0.00]*243]
	
	newimage = np.array(elist)
	newimage.resize(9,9,3)
	
	nlist = [[0.00]*243]
	goalPatch = np.array(nlist)
	goalPatch.resize(9,9,3)
	
	for i in range(row1 - 4, row1 + 5):
		for j in range(col1 -4, col1 + 5):
			for k in range(0,3):
				newimage[i-row1 + 4,j-col1 + 4,k] = im[i,j,k]
	
	for i in range(row2 - 4, row2 + 5):
		for j in range(col2 -4, col2 + 5):
			for k in range(0,3):
				goalPatch[i-row2 + 4,j-col2 + 4,k] = im2[i,j,k]
				
	tlist = [[0.0]*486]
	test = np.array(tlist)
	test.resize(9,9,6)
	#print newimage
	
	sub = newimage - goalPatch
	sub = sub*sub
	matchingCost = round(sub.sum(),5)
	return matchingCost

def comparePatches_NCC(path1, path2, row1,col1,row2,col2):
	
	im = cv2.imread(path1)/255.0
	im2 = cv2.imread(path2)/255.0
	
	
	elist = [[0.00]*243]
	
	newimage = np.array(elist)
	newimage.resize(9,9,3)
	
	nlist = [[0.00]*243]
	goalPatch = np.array(nlist)
	goalPatch.resize(9,9,3)
	
	for i in range(row1 - 4, row1 + 5):
		for j in range(col1 -4, col1 + 5):
			for k in range(0,3):
				newimage[i-row1 + 4,j-col1 + 4,k] = im[i,j,k]
	
	for i in range(row2 - 4, row2 + 5):
		for j in range(col2 -4, col2 + 5):
			for k in range(0,3):
				goalPatch[i-row2 + 4,j-col2 + 4,k] = im2[i,j,k]
				
	tlist = [[0.0]*486]
	test = np.array(tlist)
	test.resize(9,9,6)
	
		
	leftPatch = goalPatch
	rightPatch = newimage
	#NCC matching function
	l = np.mean(leftPatch)
	rr = np.mean(rightPatch)
				
	leftPatch = leftPatch - l
	rightPatch = rightPatch - rr
				
	left = leftPatch.reshape((1,-1))
	right = rightPatch.reshape((1,-1))
				
	left_norm = math.sqrt((left*left).sum())
	right_norm = math.sqrt((right*right).sum())
				
	left = left/left_norm
	right = right/right_norm
				
	right = right.reshape((-1,1))
				
	cost = np.dot(left, right)
	matchingCost = round(cost.sum(),5)
	return matchingCost

def match(im1, im2,startR, startC, scaled, deploy, caffe_model, mean_f):

	mean_file='MEAN_FILE.npy' #mean_file
	mean_npy = np.load(mean_file)
	mean = mean_npy.mean(1).mean(1)
	
	if mean_f == 1:
		net = caffe.Classifier(deploy, caffe_model, mean = mean, raw_scale = 255, channel_swap=(2,1,0,5,4,3))
	else:
		net = caffe.Classifier(deploy, caffe_model,raw_scale = 255, channel_swap=(2,1,0,5,4,3))
	net.blobs['data'].reshape(1,6,9,9)
	tC = startC
	numOfRow = 1
	numOfCol = 1
	
	left = [([0.0]*(numOfCol+startC+6)) for i in range(numOfRow+6)]
	right = [([0.0]*(numOfCol+startC+6)) for i in range(numOfRow+6)]

	path = im2
	im = cv2.imread(im1)/255.0
	im2 = cv2.imread(im2)/255.0

	if startC - 255/scaled - 1 - 4 >= 0:
		check = startC -255/scaled - 1
	else:
		check = 4
	for row in range(startR-2, startR+numOfRow+2):
		for col in range(check-2, startC+numOfCol+2):
			i = r.readIm(row, col, im, im2, 0)
			output = net.predict([i])
			left[row-startR][col] = copy.deepcopy(net.blobs['ip2'].data)#save the feature map
			right[row-startR][col] = copy.deepcopy(net.blobs['ip2_p'].data)
			
	if mean_f == 1:
		net = caffe.Classifier(deploy, caffe_model, mean = mean, raw_scale = 255, channel_swap=(2,1,0,5,4,3))
	else:
		net = caffe.Classifier(deploy, caffe_model,raw_scale = 255, channel_swap=(2,1,0,5,4,3))
		
	net.blobs['data'].reshape(1,6,9,9)
	for row in range(startR, startR+numOfRow):
		for col in range(startC, startC+numOfCol):
			d = 0
			bestDisparity = 0
			min = 100.0
			while col - 4 - d >= 0 and d <= 255/scaled:
				matchingCost = 0.0
				for i in range(row-2, row+3):
					for j in range(col - 2, col +3):
						net.blobs['ip2'].data[...] = left[row-startR][col]#assign values to this layer
						net.blobs['ip2_p'].data[...] = right[row-startR][col-d]
						pre = net.forward(start='concat', end='prob')
						output = pre['prob']#get the probability
				
						predictions = output[0]		
						matchingCost = matchingCost + predictions[0]
				
			
				if matchingCost < min:
					min = predictions[0]	
					bestDisparity = d
					tC = startC - bestDisparity
				d = d + 1
			
	list = [0,0,0,0]
	tR = startR
	list[0] = bestDisparity
	list[1] = min
	list[2] = tC
	list[3] = tR
	
	newimage = r.extractPatch(tR, tC,path,9)
	cv2.imwrite('right_patch_match.png', newimage)
	return list
	
def match_ED(im1, im2,startR, startC):
	
	tC = startC
	numOfRow = 1
	numOfCol = 1
	scaled = 1

	path = im2
	
	for row in range(startR, startR+numOfRow):
		for col in range(startC, startC+numOfCol):
			d = 0
			bestDisparity = 0
			min = 2.0
			while col - 4 - d >= 0 and d <= 255/scaled:
				matchingCost = 0.0
				
				leftPatch = r.extractPatch(row, col, im1, 1)
				rightPatch = r.extractPatch(row, col-d, im2, 1)
				sub = leftPatch - rightPatch#SSD matching function
				sub = sub*sub
				matchingCost = sub.sum()
				
				if matchingCost < min:
					min = matchingCost
					bestDisparity = d
					tC = startC - bestDisparity
				d = d + 1
			
	list = [0,0,0,0]
	tR = startR
	list[0] = bestDisparity
	list[1] = round(min,5)
	list[2] = tC
	list[3] = tR
	
	newimage = r.extractPatch(tR, tC,path, 9)
	cv2.imwrite('right_patch_match.png', newimage)
	return list
	
def match_NCC(im1, im2,startR, startC):
	
	tC = startC
	numOfRow = 1
	numOfCol = 1
	scaled = 1

	path = im2
	
	for row in range(startR, startR+numOfRow):
		for col in range(startC, startC+numOfCol):
			d = 0
			bestDisparity = 0
			max = 0.0
			while col - 4 - d >= 0 and d <= 255/scaled:
				matchingCost = 0.0
				
				leftPatch = r.extractPatch(row, col, im1, 1)
				rightPatch = r.extractPatch(row, col-d, im2, 1)
				#NCC matching function
				l = np.mean(leftPatch)
				rr = np.mean(rightPatch)
				
				leftPatch = leftPatch - l
				rightPatch = rightPatch - rr
				
				left = leftPatch.reshape((1,-1))
				right = rightPatch.reshape((1,-1))
				
				left_norm = math.sqrt((left*left).sum())
				right_norm = math.sqrt((right*right).sum())
				
				left = left/left_norm
				right = right/right_norm
				
				right = right.reshape((-1,1))
				
				cost = np.dot(left, right)
				matchingCost = cost.sum()
				
				if matchingCost > max:
					max = matchingCost
					bestDisparity = d
					tC = startC - bestDisparity
				d = d + 1
			
	list = [0,0,0,0]
	tR = startR
	list[0] = bestDisparity
	list[1] = round(max,5)
	list[2] = tC
	list[3] = tR
	
	newimage = r.extractPatch(tR, tC,path, 9)
	cv2.imwrite('right_patch_match.png', newimage)
	return list

