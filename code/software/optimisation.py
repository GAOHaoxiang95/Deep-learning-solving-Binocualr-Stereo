#author: Haoxiang Gao 25/03/2018
#This flle is the command line version of my software
#used for debuging
import os
import caffe
import numpy as np
import readImage as r
from skimage import io
import copy
caffe_root='F:/caffe-master'
deploy=caffe_root+'/PROJECT/siamese/deploy.prototxt' #deploy

caffe_model=caffe_root+'/PROJECT/siamese/model1.caffemodel'  #caffe_model
mean_file=caffe_root+'/data/PROJECT/MEAN_FILE.npy' #mean_file


mean_npy = np.load(mean_file)
mean = mean_npy.mean(1).mean(1)

filelist=[]

net = caffe.Classifier(deploy, caffe_model,mean = mean,raw_scale = 255)

net.blobs['data'].reshape(1,6,9,9)

numOfRow = 10
numOfCol = 10

scaled = 3
startR = 50
startC = 50

left = [([0.0]*(numOfCol+startC+6)) for i in range(startR+numOfRow+6)]
right = [([0.0]*(numOfCol+startC+6)) for i in range(startR+numOfRow+6)]

elist = [[0]*(numOfCol*numOfRow)]
newimage = np.array(elist)
newimage.resize(numOfRow,numOfCol,3)

im = caffe.io.load_image('F:/baby2/view1.png')
im2 = caffe.io.load_image('F:/baby2/view5.png')


	
check = max(5, startC -255/scaled - 1)

for row in range(startR-2, startR+numOfRow+2):
	for col in range(check-2, startC+numOfCol+2):
		i = r.readIm(row, col, im, im2, 0)
		output = net.predict([i])
		left[row][col] = copy.deepcopy(net.blobs['ip2'].data)
		right[row][col] = copy.deepcopy(net.blobs['ip2_p'].data)
		
	print "row", row
	print "-----------------------"
	#print right[row-startR][col]
		
#for row in range(startR, startR+numOfRow):
#	for col in range(4, startC+numOfCol):
#		print right[row-startR][col]

net = caffe.Classifier(deploy, caffe_model, mean = mean,raw_scale = 255)

net.blobs['data'].reshape(1,6,9,9)

for row in range(startR, startR+numOfRow):
	for col in range(startC, startC+numOfCol):
		d = 0
		bestDisparity = 0
		min = 12.0
		while col - 4 - d >= 0 and d <= 255/scaled:
			matchingCost = 0.0
			
			for i in range(row-2, row+3):#aggregation 5*5 window
				for j in range(col - 2, col +3):
					net.blobs['ip2'].data[...] = left[i][j]
					net.blobs['ip2_p'].data[...] = right[i][j-d]
					pre = net.forward(start='concat', end='prob')
					output = pre['prob']
			
					predictions = output[0]		
					matchingCost = matchingCost + predictions[0]
		

			
			if matchingCost < min:
				min = matchingCost	
				bestDisparity = d
		
			d = d + 1
		#print bestDisparity
		#print (dis[row][col][0]*255/4) - bestDisparity
		print 'row-', row-startR, ' col-', col-startC
		print '--------------'
		
		
		newimage[row-startR,col-startC,0] = bestDisparity
		newimage[row-startR,col-startC,1] = bestDisparity
		newimage[row-startR,col-startC,2] = bestDisparity
print newimage
io.imsave('F:/test.jpg',newimage*scaled)
newimage = newimage*scaled/255.0



