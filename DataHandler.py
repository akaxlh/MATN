import pickle
import numpy as np
from scipy.sparse import csr_matrix

# predir = 'Datasets/Tmall/backup/hr_ndcg_click/'
# predir = 'Datasets/MultiInt-ML10M/buy/'
predir = 'Datasets/yelp/click/'
trnfile = predir + 'trn_'
tstfile = predir + 'tst_'
# behs = ['pv', 'fav', 'cart', 'buy']
# behs = ['neg', 'neutral', 'pos']
behs = ['tip', 'neg', 'neutral', 'pos']

def helpInit(a, b, c):
	ret = [[None] * b for i in range(a)]
	for i in range(a):
		for j in range(b):
			ret[i][j] = [None] * c
	return ret

def LoadData():
	for i in range(len(behs)):
		beh = behs[i]
		path = trnfile + beh
		with open(path, 'rb') as fs:
			mat = (2**i)*(pickle.load(fs)!=0)
		trnMat = (mat if i==0 else trnMat + mat)
		# if i == len(behs)-1:
		# 	buyMat = 1 * (mat != 0)
	buyMat = 1 * (trnMat != 0)
	# test set
	path = tstfile + 'int'
	with open(path, 'rb') as fs:
		tstInt = np.array(pickle.load(fs))
	tstStat = (tstInt!=None)
	tstUsrs = np.reshape(np.argwhere(tstStat!=False), [-1])

	return trnMat, tstInt, buyMat, tstUsrs

def getmask(low, high, trnMat, tstUsrs, tstInt):
	cnts = np.reshape(np.array(np.sum(trnMat, axis=-1)), [-1])
	lst = list()
	for usr in tstUsrs:
		lst.append((cnts[usr], usr))
	lst.sort(key=lambda x: x[0])
	length = len(lst)
	l = int(low * length)
	r = int(high * length)
	ret = set()
	for i in range(l, r):
		ret.add(lst[i][1])
	return ret

def negSamp(tembuy, curlist):
	temsize = 1000#1000
	negset = [None] * temsize
	cur = 0
	for temcur in curlist:
		if tembuy[temcur] == 0:
			negset[cur] = temcur
			cur += 1
		if cur == temsize:
			break
	negset = np.array(negset[:cur])
	return negset

def TransMat(mat):
	user, item = mat.shape
	data = mat.data
	indices = mat.indices
	indptr = mat.indptr

	newdata = [None] * len(data)
	rowInd = [None] * len(data)
	colInd = [None] * len(data)
	length = 0

	for i in range(user):
		temlocs = indices[indptr[i]: indptr[i+1]]
		temvals = data[indptr[i]: indptr[i+1]]
		for j in range(len(temlocs)):
			rowInd[length] = temlocs[j]
			colInd[length] = i
			newdata[length] = temvals[j]
			length += 1
	if length != len(data):
		print('ERROR IN Trans', length, len(data))
		exit()
	tpMat = csr_matrix((newdata, (rowInd, colInd)), shape=[item, user])
	return tpMat

def binFind(pred, shoot):
	minn = np.min(pred)
	maxx = np.max(pred)
	l = minn
	r = maxx
	while True:
		mid = (l + r) / 2
		tem = (pred - mid) > 0
		num = np.sum(tem)
		if num == shoot or np.abs(l - r)<1e-3:
			arr = tem
			break
		if num > shoot:
			l = mid
		else:
			r = mid
	return np.reshape(np.argwhere(tem), [-1])[:shoot]
