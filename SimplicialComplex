#This set of functions allows for building a Vietoris-Rips simplicial complex from point data

import numpy as np

def euclidianDist(a,b):
  return np.linalg.norm(a - b) #euclidian distance metric

#Build neighorbood graph
def buildGraph(raw_data, metric=euclidianDist): #raw_data is a numpy array
  nodes = [x for x in range(raw_data.shape[0])] #initialize node set, reference indices from original data array
  edges = [] #initialize empty edge array
  weights = [] #initialize weight array, stores the weight (which in this case is the distance) for each edge
  eps = 3.1 #epsilon distance parameter
  for i in range(raw_data.shape[0]): #iterate through each data point
      for j in range(raw_data.shape[0]-i): #inner loop to calculate pairwise point distances
          a = raw_data[i]
          b = raw_data[j+i] #each simplex is a set (no order), hence [0,1] = [1,0]; so only store one
          if (i != j+i):
              dist = metric(a,b)
              if dist <= eps:
                  edges.append({i,j+i}) #add edge
                  weights.append([len(complex)-1,dist]) #store index and weight
  return nodes,edges,weights
  
def lower_nbrs(nodeSet, edgeSet, node):
    return {x for x in nodeSet if {x,node} in edgeSet and node > x}

def rips(nodes, edges, k):
    VRcomplex = [{n} for n in nodes]
    for e in edges: #add 1-simplices (edges)
        VRcomplex.append(e)
    for i in range(k):
        for simplex in [x for x in VRcomplex if len(x)==i+2]: #skip 0-simplices
            #for each u in simplex
            nbrs = set.intersection(*[lower_nbrs(nodes, edges, z) for z in simplex])
            for nbr in nbrs:
                VRcomplex.append(set.union(simplex,{nbr}))
    return VRcomplex
