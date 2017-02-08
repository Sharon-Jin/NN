import sys
import numpy as np
from array import array
import random
import math
import time

def main():
	t0=time.time()
	# number of hidden layers
	num=6
	lr=0.05
	maxiter=60000
	ftrain_name=sys.argv[1]
	fdev_name=sys.argv[2]
	
	t_all, x_all=readfile(ftrain_name)
	n=len(x_all)
	if n>0:
		width=len(x_all[0])
	else:
		exit()

	w_out=[random.uniform(-0.5, 0.5) for i in range(num)]
	o_out=[0]*num
	d_out=[0]*num
	
	w_hid=[np.asarray([random.uniform(-0.5, 0.5) for j in range(width)]) for i in range(num)]
	o_hid=[0]*num
	d_hid=[0]*num
	pre_error=0
	error=-1

	# train
	iter=0
	while iter<maxiter:
		iter+=1
		if time.time()-t0>130 or pre_error<error and pre_error>0:
			break
		pre_error=error

		error=0.0
		for i in range(n):
			x=x_all[i]
			t=t_all[i]
			for k in range(num):
				net=np.dot(w_hid[k],x)
				o_hid[k]=sigmoid(net)
			net=np.dot(w_out, o_hid)
			o_out=sigmoid(net)
			error+=(t-o_out)**2

			# weight update
			d_out=o_out*(1-o_out)*(t-o_out)
			for k in range(num):
				d_hid[k]=o_hid[k]*(1-o_hid[k])*w_out[k]*d_out
				w_out[k]+=lr*d_out*o_hid[k]
				for j in range(width):
					w_hid[k][j]+=lr*d_hid[k]*x[j]
		print error

	print 'TRAINING COMPLETED! NOW PREDICTING.'

	# test
	tn2,xn2=readfile(fdev_name)
	th=open('music_dev_keys.txt','rb').readlines()
	err=0.0
	for i in range(len(xn2)):
		x=xn2[i]
		for k in range(num):
			net=np.dot(w_hid[k],x)
			o_hid[k]=sigmoid(net)
		net=np.dot(w_out, o_hid)
		o_out=sigmoid(net)
		if o_out>0.5:
			label= 'yes'
		else:
			label= 'no'
		print label
		#if not(label in th[i]):
		#	err+=1
	#print err/float(len(xn2))


def sigmoid(net):
	return 1.0/(1.0+math.exp(0.0-net))


def readfile(filename):
	fin=open(filename,'rb').readlines()
	n=len(fin)-1
	attrname=fin[0].split(',')
	width=len(attrname)-1

	# load data
	tn=[0]*n
	xn=[]
	for i in range(1,n+1):
		data=fin[i].strip('\r\n').split(',')
		tn[i-1]=float(data[-1]=='yes')
		val=[0]*4
		val[0]=(float(data[0])-1900.0)/100.0
		val[1]=(float(data[1])-1.50)/5.50
		val[2]=float(data[2]=='yes')
		val[3]=float(data[3]=='yes')
		xn.append(np.asarray(val))
	return tn,xn


if __name__=='__main__':
	main()
