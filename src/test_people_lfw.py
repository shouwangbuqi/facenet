"""Helper for evaluation on the Labeled Faces in the Wild dataset 
"""

# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import facenet

def evaluate(embeddings, paths,actual_issame):
    # Calculate evaluation metrics
    paths1=paths[0::2]
    paths2=paths[1::2]
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    nrof_pairs=min(len(actual_issame),embeddings1.shape[0])
    diff = np.subtract(embeddings1,embeddings2)
    dist = np.sum(np.square(diff),1)
    indices=np.arange(nrof_pairs)
    result_list=[]
    thresholds=np.arange(0,2,0.05)
    index_truelist=np.zeros((len(thresholds),))
    index_falselist=np.zeros((len(thresholds),))
    for i in indices:
	result_list.append([paths1[i].split('/')[-1],paths2[i].split('/')[-1],dist[i],actual_issame[i]])
	if actual_issame[i] == True:
	    index_truelist[dist[i]//0.05] += 1
	else:
	    index_falselist[dist[i]//0.05] +=1
    print ("true_list is",index_truelist)
    print ("false_list is",index_falselist)
    return result_list

def get_paths(lfw_dir,people_list,file_ext):
    nrof_skipped_pairs = 0
    people_listinfo=[]
    path_list = []
    issame_list = []
    for people in people_list:
	print (people)
	path = os.path.join(lfw_dir,people[0],people[0]+'_'+'%04d' %int(people[1])+'.'+file_ext)
	name = people[0]
	people_info = (path,name)
	people_listinfo.append(people_info)
    for i in range(len(people_listinfo)):
	for j in range(len(people_listinfo)):
	    path_list+=(people_listinfo[i][0],people_listinfo[j][0])
	    if people_listinfo[i][1] == people_listinfo[j][1]:
		issame = True
	    else:
		issame =False
	    issame_list.append(issame)	
    print("path_list, issame_list ", len(path_list), len(issame_list))
    return path_list, issame_list
def read_people(people_filename):
    peoples=[]
    with open(people_filename,'r') as f:
	for line in f.readlines():
	    people = line.strip().split()
	    peoples.append(people)
    return np.array(peoples)

def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)

if __name__ == '__main__':
    path_list=[]
    issame_list=[]
    selflfw_dir='home/lf/face/datasets/self_test_160'
    path_list = read_people('/home/lf/face/facenet/data/self_people.txt')	
    path_list,issame_list=get_paths(selflfw_dir,path_list)
   
