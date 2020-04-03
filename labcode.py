import numpy as np
from Params import args
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
import Utils.NNLayers as NNs
from Utils.NNLayers import FC, Regularize, Activate, Dropout, Bias, getParam, defineParam
from DataHandler import LoadData, negSamp, binFind
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
import pickle

class Recommender:
	def __init__(self, sess, datas, inpDim):
		self.inpDim = inpDim
		self.sess = sess
		self.trnMat, self.tstInt, self.buyMat, self.tstUsrs = datas
		self.metrics = dict()
		mets = ['Loss', 'preLoss' 'HR', 'NDCG']
		for met in mets:
			self.metrics['Train'+met] = list()
			self.metrics['Test'+met] = list()

	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			tem = name + metric
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret

	def run(self):
		self.prepareModel()
		log('Model Prepared')
		if args.load_model != None:
			self.loadModel()
			stloc = len(self.metrics['TrainLoss'])
		else:
			stloc = 0
			init = tf.global_variables_initializer()
			self.sess.run(init)
			log('Variables Inited')
		for ep in range(stloc, args.epoch):
			test = (ep % 3 == 0)
			reses = self.trainEpoch()
			log(self.makePrint('Train', ep, reses, test))
			if test:
				reses = self.testEpoch()
				log(self.makePrint('Test', ep, reses, test))
			if ep % 5 == 0:
				self.saveHistory()
			print()
		reses = self.testEpoch()
		log(self.makePrint('Test', args.epoch, reses, True))
		self.saveHistory()

	def multiHeadAttention(self, localReps, glbRep, number, numHeads, inpDim):
		query = tf.reshape(tf.tile(tf.reshape(FC(glbRep, inpDim, useBias=True, reg=True), [-1, 1, inpDim]), [1, number, 1]), [-1, numHeads, inpDim//numHeads])
		temLocals = tf.reshape(localReps, [-1, inpDim])
		key = tf.reshape(FC(temLocals, inpDim, useBias=True, reg=True), [-1, numHeads, inpDim//numHeads])
		val = tf.reshape(FC(temLocals, inpDim, useBias=True, reg=True), [-1, number, numHeads, inpDim//numHeads])
		att = tf.nn.softmax(2*tf.reshape(tf.reduce_sum(query * key, axis=-1), [-1, number, numHeads, 1]), axis=1)
		attRep = tf.reshape(tf.reduce_sum(val * att, axis=1), [-1, inpDim])
		return attRep

	def selfAttention(self, localReps, number, inpDim):
		attReps = [None] * number
		stkReps = tf.stack(localReps, axis=1)
		for i in range(number):
			glbRep = localReps[i]
			temAttRep = self.multiHeadAttention(stkReps, glbRep, number=number, numHeads=args.att_head, inpDim=inpDim) + glbRep
			# fc1 = FC(temAttRep, inpDim, reg=True, useBias=True, activation='relu') + temAttRep
			# fc2 = FC(fc1, inpDim, reg=True, useBias=True, activation='relu') + fc1
			attReps[i] = temAttRep#fc2
		return attReps

	def divide(self, interaction):
		ret = [None] * self.intTypes
		for i in range(self.intTypes):
			ret[i] = tf.to_float(tf.bitwise.bitwise_and(interaction, (2**i)) / (2**i))
		return ret

	def mine(self, interaction):
		activation = 'relu'
		V = NNs.defineParam('v', [self.inpDim, args.latdim], reg=True)
		divideLst = self.divide(interaction)
		catlat1 = []
		for dividInp in divideLst:
			catlat1.append(dividInp @ V)
		catlat2 = self.selfAttention(catlat1, number=self.intTypes, inpDim=args.latdim)
		catlat3 = list()
		self.memoAtt = []
		for i in range(self.intTypes):
			resCatlat = catlat2[i] + catlat1[i]
			memoatt = FC(resCatlat, args.memosize, activation='relu', reg=True, useBias=True)
			memoTrans = tf.reshape(FC(memoatt, args.latdim**2, reg=True, name='memoTrans'), [-1, args.latdim, args.latdim])
			self.memoAtt.append(memoatt)

			tem = tf.reshape(resCatlat, [-1, 1, args.latdim])
			transCatlat = tf.reshape(tem @ memoTrans, [-1, args.latdim])
			catlat3.append(transCatlat)

		stkCatlat3 = tf.stack(catlat3, axis=1)

		weights = NNs.defineParam('fuseAttWeight', [1, self.intTypes, 1], reg=True, initializer='zeros')
		sftW = tf.nn.softmax(weights*2, axis=1)
		fusedLat = tf.reduce_sum(sftW * stkCatlat3, axis=1)
		self.memoAtt = tf.stack(self.memoAtt, axis=1)

		lat = fusedLat
		for i in range(2):
			lat = FC(lat, args.latdim, useBias=True, reg=True, activation=activation) + lat
		return lat

	def prepareModel(self):
		self.intTypes = 4
		self.interaction = tf.placeholder(dtype=tf.int32, shape=[None, self.inpDim], name='interaction')
		self.posLabel = tf.placeholder(dtype=tf.int32, shape=[None, None], name='posLabel')
		self.negLabel = tf.placeholder(dtype=tf.int32, shape=[None, None], name='negLabel')
		intEmbed = tf.reshape(self.mine(self.interaction), [-1, 1, args.latdim])
		self.learnedEmbed = tf.reshape(intEmbed, [-1, args.latdim])

		W = NNs.defineParam('W', [self.inpDim, args.latdim], reg=True)
		posEmbeds = tf.transpose(tf.nn.embedding_lookup(W, self.posLabel), [0, 2, 1])
		negEmbeds = tf.transpose(tf.nn.embedding_lookup(W, self.negLabel), [0, 2, 1])
		sampnum = tf.shape(self.posLabel)[1]

		posPred = tf.reshape(intEmbed @ posEmbeds, [-1, sampnum])
		negPred = tf.reshape(intEmbed @ negEmbeds, [-1, sampnum])
		self.posPred = posPred

		self.preLoss = tf.reduce_mean(tf.reduce_sum(tf.maximum(0.0, 1.0 - (posPred - negPred)), axis=-1))
		self.regLoss = args.reg * Regularize(method='L2')
		self.loss = self.preLoss + self.regLoss

		globalStep = tf.Variable(0, trainable=False)
		learningRate = tf.train.exponential_decay(args.lr, globalStep, args.decay_step, args.decay, staircase=True)
		self.optimizer = tf.train.AdamOptimizer(learningRate).minimize(self.loss, global_step=globalStep)

	def trainEpoch(self):
		trnMat = self.trnMat
		num = trnMat.shape[0]
		trnSfIds = np.random.permutation(num)[:args.trn_num]
		tstSfIds = self.tstUsrs
		sfIds = np.random.permutation(np.concatenate((trnSfIds, tstSfIds)))
		# sfIds = trnSfIds
		epochLoss, epochPreLoss = [0] * 2
		num = len(sfIds)
		steps = int(np.ceil(num / args.batch))

		for i in range(steps):
			curLst = list(np.random.permutation(self.inpDim))
			st = i * args.batch
			ed = min((i+1) * args.batch, num)
			batchIds = sfIds[st: ed]

			temTrn = trnMat[batchIds].toarray()
			tembuy = self.buyMat[batchIds].toarray()

			temPos = [[None]*(args.posbat*args.negsamp) for i in range(len(batchIds))]
			temNeg = [[None]*(args.posbat*args.negsamp) for i in range(len(batchIds))]
			for ii in range(len(batchIds)):
				row = batchIds[ii]
				posset = np.reshape(np.argwhere(tembuy[ii]!=0), [-1])
				negset = negSamp(tembuy[ii], curLst)
				idx = 0
				# if len(posset) == 0:
				# 	posset = np.random.choice(list(range(args.item)), args.posbat)
				for j in np.random.choice(posset, args.posbat):
					for k in np.random.choice(negset, args.negsamp):
						temPos[ii][idx] = j
						temNeg[ii][idx] = k
						idx += 1
			target = [self.optimizer, self.preLoss, self.regLoss, self.loss]
			res = self.sess.run(target, feed_dict={self.interaction: (temTrn).astype('int32'),
				self.posLabel: temPos, self.negLabel: temNeg
				}, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
			preLoss, regLoss, loss = res[1:]

			epochLoss += loss
			epochPreLoss += preLoss
			log('Step %d/%d: loss = %.2f, regLoss = %.2f       ' %\
				(i, steps, loss, regLoss), save=False, oneline=True)
		ret = dict()
		ret['Loss'] = epochLoss / steps
		ret['preLoss'] = epochPreLoss / steps
		return ret

	def testEpoch(self):
		trnMat = self.trnMat
		tstInt = self.tstInt
		epochHit, epochNdcg = [0] * 2
		ids = self.tstUsrs
		num = len(ids)
		testbatch = args.batch
		steps = int(np.ceil(num / testbatch))
		for i in range(steps):
			st = i * testbatch
			ed = min((i+1) * testbatch, num)
			batchIds = ids[st:ed]

			temTrn = trnMat[batchIds].toarray()
			temTst = tstInt[batchIds]
			tembuy = self.buyMat[batchIds].toarray()

			# get test locations
			tstLocs = [None] * len(batchIds)
			for j in range(len(batchIds)):
				negset = np.reshape(np.argwhere(tembuy[j]==0), [-1])
				rdnNegSet = np.random.permutation(negset)
				tstLocs[j] = list(rdnNegSet[:99])
				tem = ([rdnNegSet[99]] if temTst[j] in tstLocs[j] else [temTst[j]])
				tstLocs[j] = tstLocs[j] + tem

			preds = self.sess.run(self.posPred, feed_dict={self.interaction:temTrn.astype('int32'), self.posLabel: tstLocs}, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))

			hit, ndcg = self.calcRes(preds, temTst, tstLocs)
			epochHit += hit
			epochNdcg += ndcg
			log('Step %d/%d: hit = %d, ndcg = %d      ' %\
				(i, steps, hit, ndcg), save=False, oneline=True)
		ret = dict()
		ret['HR'] = epochHit / num
		ret['NDCG'] = epochNdcg / num
		return ret

	def calcRes(self, preds, temTst, tstLocs):
		hit = 0
		ndcg = 0
		for j in range(preds.shape[0]):
			predvals = list(zip(preds[j], tstLocs[j]))
			predvals.sort(key=lambda x: x[0], reverse=True)
			shoot = list(map(lambda x: x[1], predvals[:args.shoot]))
			if temTst[j] in shoot:
				hit += 1
				ndcg += np.reciprocal(np.log2(shoot.index(temTst[j])+2))
		return hit, ndcg
	
	def saveHistory(self):
		if args.epoch == 0:
			return
		with open('History/' + args.save_path + '.his', 'wb') as fs:
			pickle.dump(self.metrics, fs)

		saver = tf.train.Saver()
		saver.save(self.sess, 'Models/' + args.save_path)
		log('Model Saved: %s' % args.save_path)

	def loadModel(self):
		saver = tf.train.Saver()
		saver.restore(sess, 'Models/' + args.load_model)
		with open('History/' + args.load_model + '.his', 'rb') as fs:
			self.metrics = pickle.load(fs)
		log('Model Loaded')	

if __name__ == '__main__':
	logger.saveDefault = True
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	log('Start')
	datas = LoadData()
	log('Load Data')

	with tf.Session(config=config) as sess:
		recom = Recommender(sess, datas, args.item)
		recom.run()