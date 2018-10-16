# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 19:08:17 2018

@author: bp
"""
import numpy as np
import re
import pandas as pd
from igraph import *


def get_authors(node_info):
    #get a set with all the authors from all the papers in node_info
    authors=[]
    for i in range(len(node_info)) :   
        auth=node_info[i][3]
        auth=unif_auth(auth)
        for j in auth:
            authors.append(j)
    return set(authors)

def authors_df(authors):
    #set a pandas dataframe with all the authors in rows and columns in order to build this adjacency matrix
    return pd.DataFrame(0, index=authors, columns=authors)

def set_authors_links(training_set, authors_df):
    #get all the elements in the training set where there is a citation
    training_set2 = [element for element in training_set if element[2]=='1']
    #from this element, add +1 in the dataframe each time target_auth and source_auth have a citation
    for i in range(len(training_set2)):
        source = training_set[i][0]
        target = training_set[i][1]

        source_info = [element for element in node_info if element[0]==source][0]
        target_info = [element for element in node_info if element[0]==target][0]
        target_auth = target_info[3]
        source_auth = source_info[3]
        
        target_auth=unif_auth(target_auth)
        source_auth=unif_auth(source_auth)

        authors_df.loc[target_auth,source_auth]+=1
        authors_df.loc[source_auth,target_auth]+=1
        if (i%1000==True) :
            print (i)   
        
def get_authors_links(target_auth,source_auth, authors_df):
        #get from the adjacency matric authors_df the number of citation between all the target_auth and the source_auth
        target_auth =unif_auth(target_auth)
        source_auth =unif_auth(source_auth)
        links=authors_df.loc[target_auth,source_auth]
        
        return links.sum().sum()

def unif_auth(auth_string):
    #from the string of authors in node_info, get an array where the names are "cleaned up" and uniformized.
    #no more spaces
    #no more university names between parenthesis
    #everything in lower cases
    auth=re.sub(r'\((.*?)\)', '', auth_string)
    auth=auth.lower().split(",")
    a=[]
    for j in auth:
        j=j.split("(")
        j=j[0]
        j.strip(" ")
        while j.startswith(" "):
            j=j[1:]
        while j.endswith(" "):
            j=j[:-1]
        a.append(j)
    return a

def node_to_dic(node_info):
    #transform the node_info in dictionnary to accelerate the access to the nodes
    new_node_info={}
    for i in range (len(node_info)):
        new_node_info[node_info[i][0]]=node_info[i]
    return new_node_info

def get_graph(df, authors):
    #from the adjacency and the authors, build an undirected igraph graph with a node for each authors and the chosen distance between each nodes
    
    g=Graph()
    node_names=list(authors)
    
    # Get the values as np.array
    A = df.values
    
    # Create graph
    #igraph adjacency takes only 1 if there is an edge or 0 
    g = Graph.Adjacency((A > 0).tolist(), mode=ADJ_UNDIRECTED)
    
    # Add edge weights and node labels.
    #we add the weights of each edges and the names of the author corresponding to each node
    g.es['weight'] = sigmoid(1/(A[A.nonzero()])).tolist()
    g.vs['names'] = node_names
    return g

def dist_2_auth_str(g, authors,auth1, auth2):
    #get the distance between auth1 and auth2 in the graph
    l=list(authors)
    i=l.index(auth1)
    j=l.index(auth2)
    return dist_2_auth_int(g, i,j)

def dist_2_auth_int(g, i,j):
    #get the distance between nodes number i and j in the graph
    #get the edges of the shortest path
    es=g.get_shortest_paths(i, to=j, weights='weight', mode=ALL, output="epath")
    #then sum the weights of all the edges
    d=0
    for edge in es[0]:
        d+=g.es['weight'][edge+1]
    if d==0 :
        #if two nodes share no path, we put an arbitrary distance of 10
        d=10
    return d

def dist_all_auth(g,authors, target_auth,source_auth):
    #get the distance between all target_auth and source_auth
    #the adequate choice has not been made yet
    target_auth =unif_auth(target_auth)
    source_auth =unif_auth(source_auth)
    d=0
    for auth in target_auth:
        for auth2 in source_auth:
            d+=dist_2_auth_str(g, authors,auth, auth2)
    k=len(target_auth)*len(source_auth)
    if k==0:
        return 10
    #for now, it returns the average distance wetween the authors
    return d/k

def sigmoid(z):
    s = 1.0 / (1.0 + np.exp(-1.0 * z))
    return s
