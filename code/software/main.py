#author: Haoxiang Gao
#This is the main entrance of the program

from PyQt4 import QtGui
from PyQt4 import QtCore
from PyQt4.QtGui import *  
from PyQt4.QtCore import * 
import copy
import sys
import project as p
import readImage as r
import comparePatches as cp
import caffe
import cv2
import numpy as np
import math

class MapThread(QThread):  
    trigger = pyqtSignal() 
    bar = pyqtSignal(int)	
    def __int__(self, parent=None):  
        super(MapThread,self).__init__()  
       
  
    def run(self):
		global flag
		if flag == 1:#ssd
			elist = [[0]*(numOfCol*numOfRow)]
			newimage = np.array(elist)
			newimage.resize(numOfRow,numOfCol,3)
			
			#extract patches from left image
			list = [([0]*(numOfCol+startC+2)) for i in range(startR+numOfRow+2)]
			for i in range(startR, startR + numOfRow):
				for j in range(startC, startC + numOfCol):
					list[i][j] = r.extractPatch(i, j, lefti, 1)
				p = ((i-startR+1)/float(numOfRow))*45
				self.bar.emit(p)
			
			#extract patches from right image
			list2 = [([0]*(numOfCol+startC+2)) for i in range(startR+numOfRow+2)]
			for i in range(startR, startR + numOfRow):
				for j in range(startC, startC + numOfCol):
					list2[i][j] = r.extractPatch(i, j, righti, 1)
				p = (((i-startR+1)/float(numOfRow))*45)
				self.bar.emit(p+45)

			for row in range(startR, startR+numOfRow):
				for col in range(startC, startC+numOfCol):
					d = 0
					bestDisparity = 0
					min = 12.0
					leftPatch = list[row][col]
					while col - 4 - d >= startC and d <= 255/scaled:
					
						matchingCost = 0.0		
						rightPatch = list2[row][col-d]
						#SSD matching function
						sub = leftPatch - rightPatch
						sub = sub*sub
						matchingCost = sub.sum()
				
						if matchingCost < min:
							min = matchingCost	
							bestDisparity = d		
						d = d + 1
					
					newimage[row-startR,col-startC,0] = bestDisparity
					newimage[row-startR,col-startC,1] = bestDisparity
					newimage[row-startR,col-startC,2] = bestDisparity
				p = int(((row-startR+1)/float(numOfRow))*10)
				self.bar.emit(p+90)
			
			cv2.imwrite('disparityMap_SSD.jpg',newimage*scaled)
			newimage = newimage*scaled/255.0
			self.trigger.emit()
		elif flag == 0:#cnn
			
			global model
			global weights
			global mean_f
			deploy = model
			caffe_model = weights  #caffe_model
			
			mean_file='MEAN_FILE.npy' #mean_file

			mean_npy = np.load(mean_file)
			mean = mean_npy.mean(1).mean(1)

			if mean_f == 1:
				net = caffe.Classifier(deploy, caffe_model,mean = mean,raw_scale = 255, channel_swap=(2,1,0,5,4,3))
			else:
				net = caffe.Classifier(deploy, caffe_model,raw_scale = 255, channel_swap=(2,1,0,5,4,3))

			net.blobs['data'].reshape(1,6,9,9)
		
			left = [([0.0]*(numOfCol+startC+6)) for i in range(startR+numOfRow+6)]
			right = [([0.0]*(numOfCol+startC+6)) for i in range(startR+numOfRow+6)]

			elist = [[0]*(numOfCol*numOfRow)]
			newimage = np.array(elist)
			newimage.resize(numOfRow,numOfCol,3)

			im = cv2.imread(lefti)/255.0
			im2 = cv2.imread(righti)/255.0

				
			check = max(5, startC -255/scaled - 1)

			for row in range(startR, startR+numOfRow):
				for col in range(check, startC+numOfCol):
					i = r.readIm(row, col, im, im2, 0)
					output = net.predict([i])
					left[row][col] = copy.deepcopy(net.blobs['ip2'].data)#save features of every pixel
					right[row][col] = copy.deepcopy(net.blobs['ip2_p'].data)
				p = ((row-startR+2)/float(numOfRow))*40
				self.bar.emit(p)

			if mean_f == 1:#use mean file
				net = caffe.Classifier(deploy, caffe_model,mean = mean,raw_scale = 255, channel_swap=(2,1,0,5,4,3))
			else:
				net = caffe.Classifier(deploy, caffe_model,raw_scale = 255, channel_swap=(2,1,0,5,4,3))

			net.blobs['data'].reshape(1,6,9,9)

			for row in range(startR, startR+numOfRow):
				for col in range(startC, startC+numOfCol):
					d = 0
					bestDisparity = 0
					min = 12.0
					while col - 4 - d >= startC and d <= 255/scaled:
						matchingCost = 0.0
									
						net.blobs['ip2'].data[...] = left[row][col]
						net.blobs['ip2_p'].data[...] = right[row][col-d]
						pre = net.forward(start='concat', end='prob')
						output = pre['prob']
						
						predictions = output[0]		
						matchingCost = matchingCost + predictions[0]
									
						if matchingCost < min:
							min = matchingCost	
							bestDisparity = d
					
						d = d + 1		
					newimage[row-startR,col-startC,0] = bestDisparity
					newimage[row-startR,col-startC,1] = bestDisparity
					newimage[row-startR,col-startC,2] = bestDisparity
				p = int(((row-startR+1)/float(numOfRow))*40)
				self.bar.emit(p+60)
		
			cv2.imwrite('disparityMap_CNN.jpg', newimage*scaled)
			newimage = newimage*scaled/255.0
			print newimage
			self.trigger.emit()
			
		elif flag == 2:#NCC generate disparity map
			elist = [[0]*(numOfCol*numOfRow)]
			newimage = np.array(elist)
			newimage.resize(numOfRow,numOfCol,3)
			im = cv2.imread(lefti)/255.0
			im2 = cv2.imread(righti)/255.0
			list = [([0]*(numOfCol+startC+2)) for i in range(startR+numOfRow+2)]
			
			#extract patches from left image
			for i in range(startR, startR + numOfRow):
				for j in range(startC, startC + numOfCol):
					list[i][j] = r.extractPatch(i, j, lefti, 1)
				p = ((i-startR+1)/float(numOfRow))*45
				self.bar.emit(p)
			
			list2 = [([0]*(numOfCol+startC+2)) for i in range(startR+numOfRow+2)]
			for i in range(startR, startR + numOfRow):
				for j in range(startC, startC + numOfCol):
					list2[i][j] = r.extractPatch(i, j, righti, 1)
				p = (((i-startR+1)/float(numOfRow))*45)
				self.bar.emit(p+45)
			
			for row in range(startR, startR+numOfRow):
				for col in range(startC, startC+numOfCol):
					d = 0
					bestDisparity = 0
					maxx = 0
					leftPatch = list[row][col]
					while col - 4 - d >= startC and d <= 255/scaled:
						rightPatch = list2[row][col-d]
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
	
						if matchingCost > maxx:
							maxx = matchingCost	
							bestDisparity = d		
						d = d + 1
					
					newimage[row-startR,col-startC,0] = bestDisparity
					newimage[row-startR,col-startC,1] = bestDisparity
					newimage[row-startR,col-startC,2] = bestDisparity
				p = int(((row-startR+1)/float(numOfRow))*10)
		
				self.bar.emit(p+90)
			#print newimage
			cv2.imwrite('disparityMap_NNC.jpg',newimage*scaled)
			newimage = newimage*scaled/255.0
			self.trigger.emit()
		

		
		
		
		
class UI(QtGui.QMainWindow):
    leftPath = QtCore.QString()
    rightPath = QtCore.QString()
	
    def __init__(self):
        global numOfRow 
        global numOfCol 
        global scaled 
        global startR 
        global startC
        global mean_f
        super(UI, self).__init__()
        self.ui = p.Ui_Form()
        self.ui.setupUi(self)
        self.ui.loadLeft.clicked.connect(self.getLeft)
        self.ui.loadRight.clicked.connect(self.getRight)
        self.ui.leftPatch.clicked.connect(self.getLeftPatch)
        self.ui.pushButton.clicked.connect(self.getRightPatch)
        self.ui.pushButton.clicked.connect(self.computeCost)
        self.ui.pushButton_2.clicked.connect(self.startThread)
        self.ui.model_2.clicked.connect(self.getModel)
        self.ui.weights.clicked.connect(self.getWeights)
        self.ui.status.setText('No jobs')
        self.ui.predictRightPatch.clicked.connect(self.matchPatch)
        self.setWindowTitle('Stereo Matching')
        self.setWindowIcon(QtGui.QIcon('eye.jpg')) 
        self.ui.leftRow.setValidator(QtGui.QIntValidator(4, 5, self))
        self.ui.leftCol.setValidator(QtGui.QIntValidator(4, 5, self))
        self.ui.rightPatch_3.setValidator(QtGui.QIntValidator(4, 5, self))
        self.ui.rightCol.setValidator(QtGui.QIntValidator(4, 5, self))
        numOfRow =  50#default value
        numOfCol =  50
        scaled = 3
        startR =  50
        startC =  50
		
    def disMap(self):
		global numOfRow 
		global numOfCol 
		global scaled 
		global startR 
		global startC
		numOfRow = int(self.ui.height.text())
		numOfCol = int(self.ui.length.text())
		startR = int(self.ui.sr.text())
		startC = int(self.ui.sc.text())
		
    def setBar(self, percentage):
		self.ui.progressBar.setValue(percentage)
	
    
    def setStatus(self):
	    self.ui.status.setText('Finished')
    def startThread(self):
		global scaled
		global mean_f
		self.ui.status.setText('Generating')
		if self.ui.checkBox_2.isChecked():
			mean_f = 1
		else:
			mean_f = 0
			
		if self.ui.one.isChecked():
			scaled = 1
		elif self.ui.two.isChecked():
			scaled = 3
		elif self.ui.three.isChecked():
			scaled = 4
		else:
			scaled = 8
			
		self.mapThread = MapThread()
		self.mapThread.trigger.connect(self.setStatus)
		self.mapThread.bar.connect(self.setBar)
		
		global numOfRow 
		global numOfCol 
		global startR 
		global startC
		global flag
	
		if self.ui.checkBox.isChecked():
			startR = 7
			startC = 7
			numOfCol = ll - 14
			numOfRow = hh - 14
		else:
		
			numOfRow = int(self.ui.Height.text())
			numOfCol = int(self.ui.length.text())
			startR = int(self.ui.sr.text())
			startC = int(self.ui.lineEdit_3.text())
		if self.ui.CNN_produce.isChecked():
			flag = 0
		if self.ui.ED_produce.isChecked():
			flag = 1
		if self.ui.radioButton_3.isChecked():
			flag = 2
		self.mapThread.start()
    def getLeft(self):
        global lefti
        self.leftPath = QtGui.QFileDialog.getOpenFileName(self, 'Open file', 'c:\\',"Image files (*.ppm *.png)")
        
        self.ui.pixmap=QtGui.QPixmap()
        self.ui.pixmap.load(self.leftPath)
        lefti = unicode(self.leftPath)
        im = cv2.imread(unicode(self.leftPath))
        self.h = str(im.shape[0])
        self.l = str(im.shape[1])
        self.ui.leftSize.setText(self.h + ' X ' + self.l)
        self.ui.scene=QtGui.QGraphicsScene(self)
        item=QtGui.QGraphicsPixmapItem(self.ui.pixmap)
        self.ui.scene.addItem(item)
        self.ui.leftView.setScene(self.ui.scene)
        self.ui.leftRow.setValidator(QtGui.QIntValidator(4, int(self.h), self))
        self.ui.leftCol.setValidator(QtGui.QIntValidator(4, int(self.l), self))
		 		 
    def getRight(self):
        global righti
        self.rightPath = QtGui.QFileDialog.getOpenFileName(self, 'Open file', 'c:\\',"Image files (*.ppm *.png)")
        self.ui.pixmap=QtGui.QPixmap()
        self.ui.pixmap.load(self.rightPath)
        im = cv2.imread(unicode(self.rightPath))
        righti = unicode(self.rightPath)
        global hh
        global ll
        hh = str(im.shape[0])
        ll = str(im.shape[1])
        self.ui.lineEdit_2.setText(hh + ' X ' + ll)
        hh = im.shape[0]
        ll = im.shape[1]
        self.ui.scene=QtGui.QGraphicsScene(self)
        item=QtGui.QGraphicsPixmapItem(self.ui.pixmap)
        self.ui.scene.addItem(item)
        self.ui.rightView.setScene(self.ui.scene)
        self.ui.rightPatch_3.setValidator(QtGui.QIntValidator(4, int(self.h), self))
        self.ui.rightCol.setValidator(QtGui.QIntValidator(4, int(self.l), self))

    def getModel(self):
		global model
		a = QtGui.QFileDialog.getOpenFileName(self, 'Open file', 'c:\\',"Deploy files (*.prototxt)")
		self.ui.modelPath.setText(a)
		model = str(unicode(a))
	
    def getWeights(self):
		global weights
		b = QtGui.QFileDialog.getOpenFileName(self, 'Open file', 'c:\\',"Weight files (*.caffemodel)")
		self.ui.weightsPath.setText(b)
		weights = str(unicode(b))
    def getLeftPatch(self):
	
		row = int(self.ui.leftRow.text())
		col = int(self.ui.leftCol.text())
		patch = r.extractPatch(row, col, unicode(self.leftPath), 9)
		
		path = 'left_patch.png'
		cv2.imwrite(path,patch)
		
		self.ui.pixmap=QtGui.QPixmap()
		self.ui.pixmap.load(QtCore.QString(path))
		self.ui.scene=QtGui.QGraphicsScene(self)
		item=QtGui.QGraphicsPixmapItem(self.ui.pixmap)
		self.ui.scene.addItem(item)
		self.ui.leftPatch_2.setScene(self.ui.scene)
		
    def getRightPatch(self):		
		row = int(self.ui.rightPatch_3.text())
		col = int(self.ui.rightCol.text())
		patch = r.extractPatch(row, col, unicode(self.rightPath), 9)
		
		path = 'right_patch.png'
		cv2.imwrite(path,patch)
		
		self.ui.pixmap=QtGui.QPixmap()	
		self.ui.pixmap.load(QtCore.QString(path))
				
		self.ui.scene=QtGui.QGraphicsScene(self)
		item=QtGui.QGraphicsPixmapItem(self.ui.pixmap)
		self.ui.scene.addItem(item)
		self.ui.rightPatch_2.setScene(self.ui.scene)
		
    def computeCost(self):
        global weights
        global model
        row1 = int(self.ui.leftRow.text())
        col1 = int(self.ui.leftCol.text())
        row2 = int(self.ui.rightPatch_3.text())
        col2 = int(self.ui.rightCol.text())
        if self.ui.CNN_compare.isChecked():
			if self.ui.checkBox_2.isChecked():
				cost = cp.comparePatches(unicode(self.leftPath), unicode(self.rightPath), row1,col1,row2,col2, model, weights,1)
			else:
				cost = cp.comparePatches(unicode(self.leftPath), unicode(self.rightPath), row1,col1,row2,col2, model, weights,0)
        if self.ui.ED_campare.isChecked():
            cost = cp.comparePatches_ED(unicode(self.leftPath), unicode(self.rightPath), row1,col1,row2,col2)
        if self.ui.NCC_compare.isChecked():
            cost = cp.comparePatches_NCC(unicode(self.leftPath), unicode(self.rightPath), row1,col1,row2,col2)
        self.ui.rightCost.setText(str(cost))
        print cost
		
    def matchPatch(self):
		global weights
		global model
		global mean
		if self.ui.one.isChecked():
			scale = 1
		elif self.ui.two.isChecked():
			scale = 3
		elif self.ui.three.isChecked():
			scale = 4
		else:
			scale = 8
		row1 = int(self.ui.leftRow.text())
		col1 = int(self.ui.leftCol.text())
		if self.ui.ED_match.isChecked():
			list = cp.match_ED(unicode(self.leftPath), unicode(self.rightPath), row1, col1)
		if self.ui.NCC_match.isChecked():
			list = cp.match_NCC(unicode(self.leftPath), unicode(self.rightPath), row1, col1)
		if self.ui.CNN_match.isChecked():
			if self.ui.checkBox_2.isChecked():
				list = cp.match(unicode(self.leftPath), unicode(self.rightPath), row1, col1, scale, model, weights, 1)
			else:
				list = cp.match(unicode(self.leftPath), unicode(self.rightPath), row1, col1, scale, model, weights, 0)
		print list[0]
		print list[1]
		self.ui.disparityP.setText(QtCore.QString(str(list[0])))
		self.ui.disparityP_2.setText(QtCore.QString(str(list[1])))
		self.ui.rightRowP.setText(QtCore.QString(str(list[3])))
		self.ui.rightColP.setText(QtCore.QString(str(list[2])))
		
		self.ui.pixmap=QtGui.QPixmap()
		self.ui.pixmap.load(QtCore.QString('right_patch_match.png'))
		self.ui.scene=QtGui.QGraphicsScene(self)
		item=QtGui.QGraphicsPixmapItem(self.ui.pixmap)
		self.ui.scene.addItem(item)
		self.ui.rightPatchPredict.setScene(self.ui.scene)

		
def main():
    global w
    app = QtGui.QApplication(sys.argv)
    w = UI()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":

    main()