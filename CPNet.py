import xml.etree.ElementTree as etree
import networkx as nx
import random
import copy
import operator
import numpy as np

class CPNet:

	depGraph = nx.DiGraph()
	partialOrder= nx.DiGraph()
	originalVarsOrder={}
	originalVarsOrder_reverse={}
	
	oLegalOrder={}
	oLegalOrder_reverse={}
	
	adjMatrix=[]
	cptList=[]
	
	def __init__(self):
		self.depGraph.clear();
		self.partialOrder.clear();
		self.originalVarsOrder={};
		self.originalVarsOrder_reverse={};
		self.oLegalOrder={}
		self.oLegalOrder_reverse={}

	"""
	THIS IS NOT COMPLETE
	"""
	def randomCPNet(self,nNode,nParents):
		self.depGraph.clear();
		self.partialOrder.clear();
		for i in range(nNode):
			"""Add node to dep graph and create empty cp-table"""
			vertex=chr(i+97);
			self.depGraph.add_node(vertex, cpt={})
			
			nPar=random.randint(0,nNode);
			"""Add random edges"""
			if(i>0):
				for p in range(nPar):
					par=random.randint(0,i-1);
					print("1. Adding edge "+ str(par)+" -> "+str(i))
					####### CHECK self.depGraph.nodes()[par] e vertex BISOGNA USARE SEMPRE IL PUNTAMENTO A self.depGraph.nodes()[] o da vertici uguali
					print("1. Adding edge "+ self.depGraph.nodes()[par]+" -> "+vertex)
					self.depGraph.add_edge(self.depGraph.nodes()[par],vertex);
			
			"""Generate all permutation of parents"""
			listPermutation=list(self.perms(len(self.depGraph.predecessors(vertex))));
			print(listPermutation)
	
	def initSepFromOutcome(self, outcome, oLegalTo):
		
		self.depGraph = nx.DiGraph();
		self.partialOrder= nx.DiGraph();
		outcome_list = list(outcome)
		i=0
		#print len(outcome_list)
		keyCPT=['-']*len(outcome_list)
		oLegalTo_list=oLegalTo.split(',')
		for v in oLegalTo_list:
			self.depGraph.add_node(v, cpt={})
			
			if(outcome_list[i]=='0'): 
				p=['0','1']
			else:
				p=['1','0']
			
			self.depGraph.node[v]['cpt']["".join(keyCPT)]=p
			#print("var: "+v+" i:"+ str(i))
			#print p
			self.oLegalOrder[v]=i
			self.originalVarsOrder[v]=i
			self.oLegalOrder_reverse[i] = v
			self.originalVarsOrder_reverse[i] = v
			
			i=i+1
		
	def initComplFromOutcome(self, outcome, oLegalTo):
		
		self.initSepFromOutcome(outcome, oLegalTo)
		
		outcome_list = list(outcome)
		i=0
		oLegalTo_list=oLegalTo.split(',')
		
		n=len(oLegalTo_list)
		
		for i in range(n):
			var_from = oLegalTo_list[i]
			keyCPT=['-']*len(self.originalVarsOrder)
			
			for j in range(i+1,n):
				var_to = oLegalTo_list[j]
				self.addNONRedundantEdge(var_from, var_to,outcome_list[i])
		return 0
		
	def getReverseFromOutcome(self, outcome1, outcome2):
	
		"""
		Make a deep copy of the original cpnet
		"""
		reverse = copy.deepcopy(self)
		
		outcome1_list = list(outcome1)
		outcome2_list = list(outcome2)
		
		i=0
		n=len(outcome1_list)
		#print n
		"""
		Find the first variable where the two outcomes differ
		"""
		while(i<n and outcome1_list[i]==outcome2_list[i]): 	i=i+1
			
		v=self.originalVarsOrder_reverse[i]
		#print v
		
		for p in reverse.depGraph.node[v]['cpt']:
			#print reverse.depGraph.node[v]['cpt'][p]
			reverse.depGraph.node[v]['cpt'][p]=list(reversed(self.depGraph.node[v]['cpt'][p]))
			#print reverse.depGraph.node[v]['cpt'][p]
				
		return reverse

	def initFromFile(self, file, oLegalTo=None):
		"""
		Initialize the CP-net from an XML file using the format specify in:
		http://www.ece.iastate.edu/~gsanthan/crisner.html
		"""
		
		self.depGraph = nx.DiGraph();
		self.partialOrder= nx.DiGraph();
		self.originalVarsOrder={};
		self.originalVarsOrder_reverse={};
		self.oLegalOrder={}
		self.oLegalOrder_reverse={}
	
		"""Open xml file"""
		tree=etree.parse(file)
		"""Get xml root"""
		root=tree.getroot()
		"""Find all preference variable"""
		vars=root.findall('PREFERENCE-VARIABLE')
		"""Save variable name in listVars and build dependency graph nodes"""
		i=0;
		for v in vars:
			"""Add node to dep graph and create empty cp-table"""
			self.depGraph.add_node(v[0].text, cpt={})
			self.originalVarsOrder[v[0].text]=i
			self.originalVarsOrder_reverse[i]=v[0].text
			i+=1
		
		#print self.originalVarsOrder
		stats=root.findall('PREFERENCE-STATEMENT')
		for s in stats:
			"""Find variable preference statement refers to"""
			v=s.findall('PREFERENCE-VARIABLE')[0].text
			"""Find preference over the values"""
			p=s.findall('PREFERENCE')[0].text.split(':')
			for i in range(len(p)):
				p[i]=str(int(p[i])-1)
			"""Find dependency"""
			e=s.findall('CONDITION')
			keyCPT=['-']*len(self.originalVarsOrder)
			if(len(e)>0):
				for c in e:
					"""Split condition into variable and value"""
					condition=c.text.split('=')
					"""Add edge from dependent variable to actual variable"""
					#print('Adding edge: '+condition[0]+'->'+v)
					self.depGraph.add_edge(condition[0],v)
					#if keyCPT!='': keyCPT += ',' 
					"""Change the character of the variable using the orignal order"""
					pos=self.originalVarsOrder[condition[0]]
					#print pos
					keyCPT[pos] = str(int(condition[1])-1)
				"""Add conditional preference statement"""
				self.depGraph.node[v]['cpt']["".join(keyCPT)]=p
			else:
				"""Add preference statement"""
				self.depGraph.node[v]['cpt']["".join(keyCPT)]=p
				
		
		self.oLegalOrder=self.originalVarsOrder;
		self.oLegalOrder_reverse=self.originalVarsOrder_reverse;
				
		if oLegalTo !=None: self.forceOlegalTo(oLegalTo=oLegalTo)
		
		
	def getBestSolution(self, oLegal=False):
		"""
		Return the optimal outcome of the CP-net
		"""
		
		if oLegal and len(self.oLegalOrder)==0: raise Exception("A valid linear order must be specified.")
		"""Get list of nodes in topological order"""
		queue = nx.topological_sort(self.depGraph)
		if oLegal: usedOrder = self.oLegalOrder;
		else: usedOrder = self.originalVarsOrder;
			
		#print usedOrder
		
		"""Use a dictionary to remember which value is the best for each var"""
		assignments={};
		
		s=['-']*len(self.originalVarsOrder)
		
		"""Take for each variable the best value given parents"""
		for n in queue:
			#print 'Best value for ' + n
			keyCPT=['-']*len(self.originalVarsOrder)
			"""If var has no parents"""
			if(self.depGraph.in_degree(n)==0):
				"""Save assignement for var"""
				assignments[n]=self.depGraph.node[n]['cpt']["".join(keyCPT)][0]
				"""Add value to best solution"""
				pos = usedOrder[n]
				s[pos] = self.depGraph.node[n]['cpt']["".join(keyCPT)][0]
			else:
				keyCPT=['-']*len(self.originalVarsOrder)
				for p in self.depGraph.predecessors(n):
					#print 'Predecessor '+ p + ' of ' + n
					"""Save assignement for var"""
					#paAss= p + '=' + assignments[p]
					#keyCPT must be computed always on the original order of the variable
					pos=self.originalVarsOrder[p]
					keyCPT[pos]=assignments[p]
					#print 'Best assignment for ' + p + ' ' + assignments[p]
					
				"""Add value to best solution"""
				pos=usedOrder[n]
				s[pos] = self.depGraph.node[n]['cpt']["".join(keyCPT)][0]
				assignments[n]=self.depGraph.node[n]['cpt']["".join(keyCPT)][0]
			
		#print assignments		
		
		return "".join(s)
		
	def getPartialOrder(self, oLegal=False):
		"""
		Set the partial order induced by the CP-net, by default it uses the linear order used during generation of the CP-net. 
		If oLegal=True and another linear order to which the CP-net is O-Legal, then it returns the partial order with values permuted using
		the new linear order.
		"""
		
		if oLegal and len(self.oLegalOrder)==0: raise Exception("A valid linear order must be specified.")
		
		self.partialOrder= nx.DiGraph();
		n=len(self.depGraph.nodes());
		"""Get list of nodes in topological order"""
		#queue = nx.topological_sort(self.depGraph);
		best=self.getBestSolution(oLegal=oLegal);
		
		for i in self.perms(n):
			self.partialOrder.add_node(i, color=0);
		
		self.DFSVisit(best, oLegal=oLegal);
		
		#return nx.dfs_tree(self.depGraph)
		
	def DFSVisit(self,outcome, oLegal=False):
	
		if oLegal and len(self.oLegalOrder)==0: raise Exception("A valid linear order must be specified.")
	
		if self.partialOrder.node[outcome]['color'] !=0 :
			return;
		
		outcome_l=list(outcome);
		self.partialOrder.node[outcome]['color']=1;
		
		for i in range(len(outcome)):
			newOutcome_l=list(outcome_l);
			if outcome_l[i]=='0':
				newOutcome_l[i]='1';
			else:
				newOutcome_l[i]='0';
			
			newOutcome="".join(newOutcome_l);	
			
			"""
			if (outcome=='0111' and newOutcome=='0101'):
				print "Color new outcome: " + str(self.partialOrder.node[newOutcome]['color']);
				print "Better: " + str(self.isBetter(outcome,newOutcome,oLegal=oLegal))
			"""
			
			#if self.partialOrder.node[newOutcome]['color'] !=2 :
			if self.isBetter(outcome,newOutcome,oLegal=oLegal): self.partialOrder.add_edge(outcome,newOutcome);
			else: self.partialOrder.add_edge(newOutcome,outcome);
		
			#self.isBetter(outcome,newOutcome);
			
			self.DFSVisit(newOutcome, oLegal=oLegal);
			
		self.partialOrder.node[newOutcome]['color']=2;
	
	
	def isBetter(self, outcome1, outcome2, oLegal=False):
		"""
		Given two outcomes which differ only for one value, return:
		true if the first is better
		false otherwise
		"""
	
		if oLegal and len(self.oLegalOrder)==0: raise Exception("A valid linear order must be specified.")
	
		"""Get list of nodes in topological order"""
		queue = nx.topological_sort(self.depGraph);
		if oLegal: 
			usedOrder = self.oLegalOrder;
			usedOrder_reverse = self.oLegalOrder_reverse;
		else: 
			usedOrder = self.originalVarsOrder;
			usedOrder_reverse = self.originalVarsOrder_reverse;
		
		i=0
		for i in range(len(outcome1)):
			if outcome1[i]!=outcome2[i]: break;
		
		#print outcome1
		#print outcome2
		
		best=False
		
		
		#print "Position that differs: "+str(i)
		"""Find which variable differs in the two outcomes"""
		varD=usedOrder_reverse[i]
		#print "Variable that differs: "+varD
		"""Retrive parents for the variable"""
		parents=self.depGraph.predecessors(varD);
		
		#print list(parents)
		#print list(parents)
		keyCPT_list=['-']*len(self.originalVarsOrder)
		#print list(parents)
		#print len(list(self.depGraph.predecessors(varD)))
		"""If variable is not independent then build the key of the parents"""
		if len(list(self.depGraph.predecessors(varD)))!=0:
			"""Since only 1 value can change all the others are equal"""
			#print "here1"
			for p in self.depGraph.predecessors(varD):
				pos=self.originalVarsOrder[p]
				keyCPT_list[pos] = outcome1[pos]
				#print keyCPT_list
			
		keyCPT="".join(keyCPT_list)
		
		#print varD
		#print keyCPT
		
		if outcome1[i]==self.depGraph.node[varD]['cpt'][keyCPT][0]: best=True
				
		return best;
		
		
	def perms(self,n):
		if not n:
			return

		for i in range(2**n):
			s = bin(i)[2:]
			s = "0" * (n-len(s)) + s
			yield s

	def isOlegal(self, order):
		""" 
		Given a linear order of the variables returns
		True if the CP-net is O-legal to the linear order
		False otherwise
		"""

		order_list=order.split(",")
		
		#print order_list
		
		i=0;
		for i in range(len(self.originalVarsOrder)):
			self.oLegalOrder[order_list[i]] = i
			self.oLegalOrder_reverse[i] = order_list[i]
	
			for j in range(i+1, len(self.originalVarsOrder)):
				if order_list[j] in self.depGraph.predecessors(order_list[i]):
					#print order_list[j] + " is a predecessor of " + order_list[i]
					return False;
		
		return True;
	
	def compareOutcomes(self, o1, o2):	
		"""
		Given two outcomes o1 and o2 returns where they are incomparable or not.
		
		Returns
        -------
			-1: the two outcomes are incomparable
			0: o1 is better than o2
			1: o2 is better than o1
		"""
		#if not (nx.has_path(self.partialOrder,o1,o2)) and not (nx.has_path(self.partialOrder,o2,o1)):
		#	return -1
		
		if nx.has_path(self.partialOrder,o1,o2):
			return 0
			
		if nx.has_path(self.partialOrder,o2,o1):
			return 1
		
		return -1
		
	def countIncomparable(self, verbose=False):
		"""
		Return the number of incomparable pair in the partial order.
		"""
		i=0
		n=len(self.partialOrder.nodes())
		listOutcomes = list(self.partialOrder.nodes())
		count=0
		for i in range(n):
			for j in range(i+1,n):
				if self.compareOutcomes(listOutcomes[i],listOutcomes[j])==-1: 
					if verbose: print(listOutcomes[i]+ " " +listOutcomes[j])
					count+=1
			
		return count;
		
	def getReverse(self):
		"""
		Return the CPNet with the same dep. graph but all the cp-entries reversed
		"""
		reverse=copy.deepcopy(self)
		for n in reverse.depGraph.nodes():
			for s in reverse.depGraph.node[n]['cpt']:
				temp=reverse.depGraph.node[n]['cpt'][s][0]
				reverse.depGraph.node[n]['cpt'][s][0]=reverse.depGraph.node[n]['cpt'][s][1]
				reverse.depGraph.node[n]['cpt'][s][1]=temp
		return reverse
	
	def addRedundantEdge(self,n1,n2):
		"""
		Add a redundant edge from n1 to n2, if n1 is not n2 and n1 is not a predecessor of n2.
		The CP-entries of n2 are duplicated in such a way that they are indifferent to the new parent.
		"""
		if n1==n2: return -1;
		
		"""
		Prevent to introduce cycles
		"""
		if nx.has_path(self.depGraph, n2, n1): return -2
		
		if n1 in self.depGraph.predecessors(n2): return -3;
	
		self.depGraph.add_edge(n1,n2);
		new_CPT={}
		
		for s in self.depGraph.node[n2]['cpt']:
			"""Save the original conditional preference statement"""
			temp=self.depGraph.node[n2]['cpt'][s]
			#list_to_del.append(s)
			
			"""Get the original CPT key"""
			keyCPT_list=list(s)
			pos=self.originalVarsOrder[n1]
			
			keyCPT_list[pos] = '0'
			keyCPT="".join(keyCPT_list)
			new_CPT[keyCPT]=temp
			keyCPT_list[pos] = '1'
			keyCPT="".join(keyCPT_list)
			new_CPT[keyCPT]=temp
			
		self.depGraph.node[n2]['cpt']=new_CPT
		return 0
	
	def addNONRedundantEdge(self,n1,n2,best):
		"""
		Add a NON redundant edge from n1 to n2, if n1 is not n2 and n1 is not a predecessor of n2.
		The CP-entries of n2 are duplicated in such a way that they are indifferent to the new parent.
		"""
		if n1==n2: return;
		
		if n1 in self.depGraph.predecessors(n2): return;
	
		self.depGraph.add_edge(n1,n2);
		new_CPT={}
		
		for s in self.depGraph.node[n2]['cpt']:
			"""Save the original conditional preference statement"""
			temp=self.depGraph.node[n2]['cpt'][s]
			#list_to_del.append(s)
			
			"""Get the original CPT key"""
			keyCPT_list=list(s)
			pos=self.originalVarsOrder[n1]
			
			keyCPT_list[pos] = '0'
			keyCPT="".join(keyCPT_list)
			if best=='0': new_CPT[keyCPT]=temp
			else: new_CPT[keyCPT]= list(reversed(temp))
			
			keyCPT_list[pos] = '1'
			keyCPT="".join(keyCPT_list)
			if best=='1': new_CPT[keyCPT]=temp
			else: new_CPT[keyCPT]= list(reversed(temp))
			
		self.depGraph.node[n2]['cpt']=new_CPT

	
	def forceOlegalTo(self, oLegalTo):
		depGraph = nx.DiGraph();
		partialOrder= nx.DiGraph();
		oLegalOrder={};
		topologicalOrder={};
		oLegalOrder_reverse={};
		
		order_list=oLegalTo.split(",")
		topological=nx.topological_sort(self.depGraph);
	
		i=0;
		for v in order_list:
			"""Add node to dep graph and create empty cp-table"""
			depGraph.add_node(v, cpt={})
			oLegalOrder[v]=i
			topologicalOrder[topological[i]]=i
			oLegalOrder_reverse[i]=v
			i+=1
	
		
		
		i=0
		for i in range(len(topological)):
			
			new_var=oLegalOrder_reverse[i]
			#print "Change "+topological[i]+" in "+new_var
			
			for p in self.depGraph.predecessors(topological[i]):
				#Get the original position of the parent in the topological order
				orig_pos=topologicalOrder[p]
				#Get the new name of the parent
				new_parent=oLegalOrder_reverse[orig_pos]
				#Add edge from the new parent
				#print "Edge from "+new_parent+ " to " + new_var
				depGraph.add_edge(new_parent, new_var)
				
			for k in self.depGraph.node[topological[i]]['cpt']:
				keyCPT_list=['-']*len(self.originalVarsOrder)
				#Get the original statement
				s=self.depGraph.node[topological[i]]['cpt'][k]
				oldKeyCPT_list=list(k)
				
				j=0
				n=len(oldKeyCPT_list)
				for j in range(n):
					#Get the original name of the variable in position j
					orig_var=self.originalVarsOrder_reverse[j]
					#Get the position of the var in the topological order
					orig_pos=topologicalOrder[orig_var]
					#Permute the keyCPT from old to new order
					keyCPT_list[orig_pos]=oldKeyCPT_list[j]
					
				depGraph.node[new_var]['cpt']["".join(keyCPT_list)]=s
				
		self.depGraph = depGraph
		self.partialOrder= nx.DiGraph();
		self.originalVarsOrder=oLegalOrder;
		self.originalVarsOrder_reverse=oLegalOrder_reverse;
		self.oLegalOrder=oLegalOrder
		self.oLegalOrder_reverse=oLegalOrder_reverse
			
	
	def getNormalized(self, cpnet):
		normalized=copy.deepcopy(self)
		
		for n in self.originalVarsOrder:
			for p in cpnet.depGraph.predecessors(n):
				normalized.addRedundantEdge(p,n)
		
		return normalized
	
	'''
	Return the shortest path from the best solution to the specified outcome
	'''
	def getDistFromBest(self,outcome):
		best=self.getBestSolution()
		path=nx.shortest_path(self.partialOrder,source=best,target=outcome)
		return len(path)-1
		
	'''
	Return a list of outcomes sorted by distance from the best solution
	'''
	def getOutcomesByDist(self):
		#best=self.getBestSolution()
		
		dict={}
		for o in self.partialOrder.nodes():
			dist=self.getDistFromBest(o)
			dict[o]=dist
		
		sorted_x = sorted(dict.items(), key=operator.itemgetter(1))
		
		return sorted_x
		
	def getAdjMatrix(self):
		"""
		Return the adjacency_matrix which represent the dependency graph of the CP-net
		Variables are ordered using the files order
		"""
		if self.adjMatrix==[]:
			order = [self.oLegalOrder_reverse[i] for i in  self.oLegalOrder_reverse]
			m = nx.adjacency_matrix(self.depGraph,order)
			self.adjMatrix=np.asarray(m.todense())
			
		return self.adjMatrix
	
	def getCPTList(self):
		
		if self.cptList==[]:
			dim=len(self.oLegalOrder_reverse)
			lst = np.zeros((pow(2,dim),dim+1))
			i=0
			for i in self.oLegalOrder_reverse:
				v = self.oLegalOrder_reverse[i]
				for key in self.depGraph.node[v]['cpt']:
					b=key+self.depGraph.node[v]['cpt'][key][0]
					b=[ord(c) for c in b]
					lst[i]=b
					i=i+1
			#lst = np.asarray(lst)
			#lst = np.reshape(lst, ( -1, len(self.originalVarsOrder)+1, 1)) 
			#print len(self.originalVarsOrder)
			#print lst.shape
			self.cptList=lst
		
		return self.cptList