from CPNet import CPNet
from scipy.special import binom
from Utils import getDir
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
#import thread
#from threading import Thread
from multiprocessing import Process, Value, Lock
import traceback

		
class Distances(object):
	cpnet1=CPNet();
	cpnet2=CPNet();
	
	kt=0.0
	gen=0.0
	cpd=0.0
	norm=0

	def __init__(self):
		self.cpnet1=CPNet();
		self.cpnet2=CPNet();
		self.ktMP = Value('f', 0)
		self.normMP = Value('i', 0)
		self.lock = Lock()
	
	def setNets(self, file1, file2, orderOlegal):
		self.cpnet1.initFromFile(file1);
		self.cpnet1.isOlegal(orderOlegal)
		self.cpnet2.initFromFile(file2);
		self.cpnet2.isOlegal(orderOlegal)
		self.cpnet1.getPartialOrder(oLegal=True)
		self.cpnet2.getPartialOrder(oLegal=True)
	
	def setNets(self,C1,C2, computePO=True):
		self.cpnet1=C1
		self.cpnet2=C2
		if computePO==True:
			self.cpnet1.getPartialOrder()
			self.cpnet2.getPartialOrder()
	"""
	def compareOutcomes(self, o1, o2):
		
		if !nx.has_path(cpnet1)
	"""
	def distKT(self,p=0.5, normalized=False, verbose=False):
		i=0
		n=len(self.cpnet1.partialOrder.nodes())
		listOutcomes = list(self.cpnet1.partialOrder.nodes())
		
		count=0
		
		dist=0.0
		self.norm=0
		
		for i in range(n):
			for j in range(i+1,n):
				o1=listOutcomes[i]
				o2=listOutcomes[j]
				r1=self.cpnet1.compareOutcomes(o1,o2)
				r2=self.cpnet2.compareOutcomes(o1,o2)
				
				if (r1==0 and r2==1) or (r1==1 and r2==0): dist +=1
				if (r1==-1 and r2!=-1) or (r1!=-1 and r2==-1): dist +=p
				if not(r1==-1 and r2==-1): self.norm += 1
				
				if verbose:
					print(getDir(r1,o1,o2) + "\t" + getDir(r2,o1,o2))
		
		self.kt=dist		
		return dist
	
	def incrementNormMP(self):
		with self.lock:
			self.normMP.value += 1

	def incrementKTMP(self, delta):
		with self.lock:
			self.ktMP.value += delta
			
	def resetNormMP(self):
		with self.lock:
			self.normMP.value = 0

	def resetKTMP(self):
		with self.lock:
			self.ktMP.value = 0
		
	def getNormMP(self):
		with self.lock:
			return self.normMP.value

	def getKTMP(self):
		with self.lock:
			return self.ktMP.value
	
	def distKT_multithread(self,p=0.5, nThreads=10, verbose=False):
	
		'''
		Each pair of outcome has an index in 0..n*n where n is the number of outcomes. Retrieve the index of the first outcome comuting
		i1=index/n and the index of the second outcome i2=index%n. Compute distance if i2>i1 to avoid computing two times the same contribution.
		Give to each thread a range of indexes that has the same mount of pairs useful to compute the distance
		'''
	
		self.kt=0.0
		self.norm=0
		i=0
		n=len(self.cpnet1.partialOrder.nodes())
		listOutcomes = list(self.cpnet1.partialOrder.nodes())
		outcomesPerChunk=(n*n)/nThreads
		lastChunk=(n*n)%nThreads
		total=(n*n)
		threads = []
		startIndex=0
		stopIndex=0
		
		try:
			for i in range(nThreads):
				count=0
				'''
				Compute the startIndex and stopIndex based on the number of useful pairs in the range to compute the distance
				'''
				j=startIndex
				while(j<total and count<outcomesPerChunk):
					o1=j/n
					o2=j%n
					if(o2>o1): count+=1
					stopIndex=j
					j+=1
				
				print("Crearing thread: " + str(i))
				temp = Process( target = compareChunk, args=(listOutcomes, n, startIndex, stopIndex, self, p ) )
				print("After creating thread: " + str(i))
				threads.append(temp)
				
				startIndex=stopIndex
		
			if lastChunk != 0: 
				temp = Process( target =  compareChunk, args=(listOutcomes, n, startIndex, n*n, self, p) )
				threads.append(temp)
			
			for t in threads:
				t.start()
				
			for t in threads:
				t.join()
		except Exception as e:	
			traceback.format_exc()
			raise e
		
		self.norm=self.getNormMP()
		self.kt=self.getKTMP()
		return self.getKTMP()
	'''
		Compute KT only for subset of outcomes
	'''
				
	
	def getKTNorm(self):
		return self.kt/self.norm
		
	def getKT(self):
		return self.kt
		
	def distGen(self):
		dist=0.0
		m=len(self.cpnet1.oLegalOrder)
		nSwap=0.0
		maxSwap=0.0
		
		c1_norma = self.cpnet1.getNormalized(self.cpnet2)
		c2_norma = self.cpnet2.getNormalized(self.cpnet1)
		
		#print c1_norma.depGraph.node['x1']
		#print c1_norma.depGraph.node['x2']

		#print c2_norma.depGraph.node['x1']
		#print c2_norma.depGraph.node['x2']
		
		count=0
		total=0
		for v in c1_norma.oLegalOrder:
			for k in c1_norma.depGraph.node[v]['cpt']:
				total +=1
				adding=pow(2,m-1-len(c2_norma.depGraph.predecessors(v))+len(c2_norma.depGraph.successors(v)))
				maxSwap += adding
				if c1_norma.depGraph.node[v]['cpt'][k]!=c2_norma.depGraph.node[v]['cpt'][k]:
					#print 'Variable ' + v + " differs " +str(c1_norma.depGraph.node[v]['cpt'][k])+" " + str(c2_norma.depGraph.node[v]['cpt'][k])
					#nSwap += pow(2,m-1-c1_norma.oLegalOrder[v]+m-1+len(c2_norma.depGraph.predecessors(v)))
					#nSwap += pow(2,m-1-c1_norma.oLegalOrder[v]+m-1-len(c2_norma.depGraph.predecessors(v)))
					nSwap += adding
					#count+=1
			#maxSwap += pow(2,c1_norma.oLegalOrder[v])
	
		#maxSwap *= total*pow(2,m-1)
		#maxSwap = total*pow(2,m-1)*(pow(2,m)-1)
		#maxSwap = pow(2,m-1)*(pow(2,m)-1)
		#maxSwap = n*pow(2,m-1)
		
		#print str(nSwap) + " " + str(maxSwap)
		
		#dist = count*nSwap/maxSwap
		dist = nSwap/maxSwap
		
		self.gen=dist
		return dist

	def distCPD(self):
		dist=0.0
		m=len(self.cpnet1.oLegalOrder)
		nSwap=0.0
		maxSwap=0.0
		
		c1_norma = self.cpnet1.getNormalized(self.cpnet2)
		c2_norma = self.cpnet2.getNormalized(self.cpnet1)
		
		#print c1_norma.depGraph.node['x1']
		#print c1_norma.depGraph.node['x2']

		#print c2_norma.depGraph.node['x1']
		#print c2_norma.depGraph.node['x2']
		
		count=0
		total=0
		for v in c1_norma.oLegalOrder:
			for k in c1_norma.depGraph.node[v]['cpt']:
				total +=1
				if c1_norma.depGraph.node[v]['cpt'][k]!=c2_norma.depGraph.node[v]['cpt'][k]:
					#print 'Variable ' + v + " differs " +str(c1_norma.depGraph.node[v]['cpt'][k])+" " + str(c2_norma.depGraph.node[v]['cpt'][k])
					#nSwap += pow(2,m-1-c1_norma.oLegalOrder[v]+m-1+len(c2_norma.depGraph.predecessors(v)))
					nSwap += pow(2,m-1-c1_norma.oLegalOrder[v]+m-1-len(c2_norma.depGraph.predecessors(v)))
					count+=1
			#maxSwap += pow(2,c1_norma.oLegalOrder[v])
	
		#maxSwap *= total*pow(2,m-1)
		#maxSwap = total*pow(2,m-1)*(pow(2,m)-1)
		maxSwap = pow(2,m-1)*(pow(2,m)-1)
		
		#print str(nSwap) + " " + str(maxSwap)
		
		#dist = count*nSwap/maxSwap
		dist = nSwap/maxSwap
		
		self.cpd=dist
		return dist

		
	def cosSim(self):
		dist=0.0
		m=len(self.cpnet1.oLegalOrder)
		nSwap=0.0
		maxSwap=0.0
		
		c1_norma = self.cpnet1.getNormalized(self.cpnet2)
		c2_norma = self.cpnet2.getNormalized(self.cpnet1)
		
		c1_vec=[]
		c2_vec=[]
		
		for v in c1_norma.oLegalOrder:
			for e in c1_norma.depGraph.node[v]['cpt']:
				c1_vec.append(1)
				if c1_norma.depGraph.node[v]['cpt'][e][0] == c2_norma.depGraph.node[v]['cpt'][e][0] : c2_vec.append(1)
				else: c2_vec.append(-1)
		
		return cosine_similarity(c1_vec,c2_vec)
	
	def pairWise(self,metric='euclidean'):
		dist=0.0
		m=len(self.cpnet1.oLegalOrder)
		nSwap=0.0
		maxSwap=0.0
		
		c1_norma = self.cpnet1.getNormalized(self.cpnet2)
		c2_norma = self.cpnet2.getNormalized(self.cpnet1)
		
		c1_vec=[]
		c2_vec=[]
		
		for v in c1_norma.oLegalOrder:
			for e in c1_norma.depGraph.node[v]['cpt']:
				c1_vec.append(1)
				if c1_norma.depGraph.node[v]['cpt'][e][0] == c2_norma.depGraph.node[v]['cpt'][e][0] : c2_vec.append(1)
				else: c2_vec.append(-1)
		
		return pairwise_distances(c1_vec,c2_vec, metric)	
		
	
def compareChunk(listOutcomes,n,startIndex,stopIndex, distanceObj, p=0.5, verbose=False):
	
	dist=0.0
	'''
	for i in range(startIndex,stopIndex):
		for j in range(i+1,n):
			o1=listOutcomes[i]
			o2=listOutcomes[j]
			r1=self.cpnet1.compareOutcomes(o1,o2)
			r2=self.cpnet2.compareOutcomes(o1,o2)
			
			if (r1==0 and r2==1) or (r1==1 and r2==0): dist +=1
			if (r1==-1 and r2!=-1) or (r1!=-1 and r2==-1): dist +=p
			if not(r1==-1 and r2==-1): self.norm += 1
			
			if verbose:
				print getDir(r1,o1,o2) + "\t" + getDir(r2,o1,o2)
	'''
	print("Beginning ")
	for i in range(startIndex,stopIndex):
		
		i1=i/n
		i2=i%n
		if(i2>i1):
			o1=listOutcomes[i1]
			o2=listOutcomes[i2]
			r1=distanceObj.cpnet1.compareOutcomes(o1,o2)
			r2=distanceObj.cpnet2.compareOutcomes(o1,o2)
			
			if (r1==0 and r2==1) or (r1==1 and r2==0): distanceObj.incrementKTMP(1)
			if (r1==-1 and r2!=-1) or (r1!=-1 and r2==-1): distanceObj.incrementKTMP(p)
			if not(r1==-1 and r2==-1): distanceObj.incrementNormMP()
	
	print("Stopping")