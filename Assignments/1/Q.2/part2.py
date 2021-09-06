import math
import numpy as np
from collections import Counter
# Note: please don't add any new package, you should solve this problem using only the packages above.
#-------------------------------------------------------------------------
'''
    Part 1: Decision Tree (with Discrete Attributes) -- 60 points --
    In this problem, you will implement the decision tree method for classification problems.
    You could test the correctness of your code by typing `nosetests -v test1.py` in the terminal.
'''
        
#-----------------------------------------------
class Node:
    '''
        Decision Tree Node (with discrete attributes)
        Inputs: 
            X: the data instances in the node, a numpy matrix of shape p by n.
               Each element can be int/float/string.
               Here n is the number data instances in the node, p is the number of attributes.
            Y: the class labels, a numpy array of length n.
               Each element can be int/float/string.
            i: the index of the attribute being tested in the node, an integer scalar 
            C: the dictionary of attribute values and children nodes. 
               Each (key, value) pair represents an attribute value and its corresponding child node.
            isleaf: whether or not this node is a leaf node, a boolean scalar
            p: the label to be predicted on the node (i.e., most common label in the node).
    '''
    def __init__(self,X,Y, i=None,C=None, isleaf= False,p=None):
        self.X = X
        self.Y = Y
        self.i = i
        self.C= C
        self.isleaf = isleaf
        self.p = p

#-----------------------------------------------
class Tree(object):
    '''
        Decision Tree (with discrete attributes). 
        We are using ID3(Iterative Dichotomiser 3) algorithm. So this decision tree is also called ID3.
    '''
    #--------------------------
    @staticmethod
    def entropy(Y):
        '''
            Compute the entropy of a list of values.
            Input:
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                e: the entropy of the list of values, a float scalar
            Hint: you could use collections.Counter.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        List=list(Y)
        occurences=dict((label,List.count(label)) for label in set(List))
        e=0
        length=len(Y)
        n=len(occurences)
        if(n>1):
            for c in occurences.values():
                e+=-(c/length)*math.log(c/length,2)
        #########################################
        return e 
    
    
            
    #--------------------------
    @staticmethod
    def conditional_entropy(Y,X):
        '''
            Compute the conditional entropy of y given x. The conditional entropy H(Y|X) means average entropy of children nodes, given attribute X. Refer to https://en.wikipedia.org/wiki/Information_gain_in_decision_trees
            Input:
                X: a list of values , a numpy array of int/float/string values. The size of the array means the number of instances/examples. X contains each instance's attribute value. 
                Y: a list of values, a numpy array of int/float/string values. Y contains each instance's corresponding target label. For example X[0]'s target label is Y[0]
            Output:
                ce: the conditional entropy of y given x, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        y=list(Y)
        x=list(X)
        ce=0
        List=[]
        for att in set(x):
            pos = [i for i, x in enumerate(x) if x == att]
            for pos1 in pos:
                List.append(y[pos1])
            e=Tree.entropy(List)
            List.clear()
            ce+=len(pos)/len(x)*e    
        #########################################
        return ce 
    
    
    
    #--------------------------
    @staticmethod
    def information_gain(Y,X):
        '''
            Compute the information gain of y after spliting over attribute x
            InfoGain(Y,X) = H(Y) - H(Y|X) 
            Input:
                X: a list of values, a numpy array of int/float/string values.
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                g: the information gain of y after spliting over x, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        ce=Tree.conditional_entropy(Y,X)
        e=Tree.entropy(Y)
        g=e-ce
        #########################################
        return g
    '''
    def split_information(Y,X):
        #Compute the split information gain
        y=list(Y)
        x=list(X)
        si=0
        List=[]
        for att in set(x):
            pos = [i for i, x in enumerate(x) if x == att]
            si+=-len(pos)/len(x)*math.log(len(pos)/len(x),2)
        #########################################
        return si 

    def gain_ratio(Y,X):
        #Compute the gain ratio 
        ig=Tree.information_gain(Y,X)
        si=Tree.split_information(Y,X)
        if(si==0):
            if(ig==0):
                return ig
            else:
                return 100
        else:
            return ig

    #--------------------------
    '''
    @staticmethod
    def best_attribute(X,Y):
        '''
            Find the best attribute to split the node. 
            Here we use information gain to evaluate the attributes. 
            If there is a tie in the best attributes, select the one with the smallest index.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
            Output:
                i: the index of the attribute to split, an integer scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        i=0
        maxg=0
        for j in range(0,np.shape(X)[0]):
            g=Tree.information_gain(Y,X[j])
            if(g>maxg):
                maxg=g
                i=j

        #########################################
        return i

        
    #--------------------------
    @staticmethod
    def split(X,Y,i):
        '''
            Split the node based upon the i-th attribute.
            (1) split the matrix X based upon the values in i-th attribute
            (2) split the labels Y based upon the values in i-th attribute
            (3) build children nodes by assigning a submatrix of X and Y to each node
            (4) build the dictionary to combine each  value in the i-th attribute with a child node.
    
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                i: the index of the attribute to split, an integer scalar
            Output:
                C: the dictionary of attribute values and children nodes. 
                   Each (key, value) pair represents an attribute value and its corresponding child node.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        x=list(X[i])
        C={}
        for att in set(x):
            
            mx=[]
            my=[]
            pos = [i for i, x in enumerate(x) if x == att]
            for j in range(0,np.shape(X)[0]):
                for pos1 in pos:
                    mx.append(X[j,pos1])
                    if(j==0):
                        my.append(Y[pos1])
            mx=np.reshape(mx,(np.shape(X)[0],len(pos)))
            my=np.reshape(my,(len(pos),))
            cnode=Node(mx,my)
            C[att]=cnode
        #########################################
        return C

    #--------------------------
    @staticmethod
    def stop1(Y):
        '''
            Test condition 1 (stop splitting): whether or not all the instances have the same label. 
    
            Input:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
            Output:
                s: whether or not Conidtion 1 holds, a boolean scalar. 
                True if all labels are the same. Otherwise, false.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        y=list(Y)
        if y.count(Y[0])==len(Y):
            s=True
        else:
            s=False
        #########################################
        return s
    
    #--------------------------
    @staticmethod
    def stop2(X):
        '''
            Test condition 2 (stop splitting): whether or not all the instances have the same attribute values. 
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
            Output:
                s: whether or not Conidtion 2 holds, a boolean scalar. 
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        s=True
        for j in range(0,np.shape(X)[0]):
            if(Tree.stop1(X[j])!=True):
                s=False
                break
        #########################################
        return s
    
            
    #--------------------------
    @staticmethod
    def most_common(Y):
        '''
            Get the most-common label from the list Y. 
            Input:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node.
            Output:
                y: the most common label, a scalar, can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        y=Counter(Y)
        y=y.most_common(1)[0][0] 
        #########################################
        return y
    
    
    
    #--------------------------
    @staticmethod
    def build_tree(t):
        '''
            Recursively build tree nodes.
            Input:
                t: a node of the decision tree, without the subtree built.
                t.X: the feature matrix, a numpy float matrix of shape p by n.
                   Each element can be int/float/string.
                    Here n is the number data instances, p is the number of attributes.
                t.Y: the class labels of the instances in the node, a numpy array of length n.
                t.C: the dictionary of attribute values and children nodes. 
                   Each (key, value) pair represents an attribute value and its corresponding child node.
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        if(Tree.stop1(t.Y)==True or Tree.stop2(t.X)==True):
            t.p=Tree.most_common(t.Y)
            t.isleaf=True
        else:
            t.p=Tree.most_common(t.Y)
            t.i=Tree.best_attribute(t.X,t.Y)
            t.C=Tree.split(t.X,t.Y,t.i)
            for value in t.C.values():
                print(t.C)
                Tree.build_tree(value)
        
        #########################################
    
    
    #--------------------------
    @staticmethod
    def train(X, Y):
        '''
            Given a training set, train a decision tree. 
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the training set, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
            Output:
                t: the root of the tree.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        t=Node(X,Y)
        Tree.build_tree(t)
        #########################################
        return t
    
    
    
    #--------------------------
    @staticmethod
    def inference(t,x):
        '''
            Given a decision tree and one data instance, infer the label of the instance recursively. 
            Input:
                t: the root of the tree.
                x: the attribute vector, a numpy vectr of shape p.
                   Each attribute value can be int/float/string.
            Output:
                y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        x=list(x)
        if(t.isleaf==True):
            return t.p
        else:
            count=0
            for att in x:
                if att in t.C:
                    x.remove(att)
                    if(t.C[att].isleaf==True):
                        return t.C[att].p
                    else:
                        return Tree.inference(t.C[att],x)
                else:
                    count+=1
            if(count==len(x)):
                return t.p        
        #########################################
        
        

    #--------------------------
    @staticmethod
    def predict(t,X):
        '''
            Given a decision tree and a dataset, predict the labels on the dataset. 
            Input:
                t: the root of the tree.
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the dataset, p is the number of attributes.
            Output:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        Y=[]
        for i in range(np.shape(X)[1]):
            Y.append(Tree.inference(t,X[:,i]))
        Y=np.reshape(Y,(np.shape(X)[1],))
        #########################################
        return Y



    #--------------------------
    @staticmethod
    def load_dataset(filename = 'credit.csv'):
        '''
            Load dataset 1 from the CSV file: 'data1.csv'. 
            The first row of the file is the header (including the names of the attributes)
            In the remaining rows, each row represents one data instance.
            The first column of the file is the label to be predicted.
            In remaining columns, each column represents an attribute.
            Input:
                filename: the filename of the dataset, a string.
            Output:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the dataset, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        dataset = np.genfromtxt(filename, delimiter = ",", dtype=None, encoding='utf-8')
        print(dataset)
        X=dataset[1:,2:]
        X=np.transpose(X)
        Y=dataset[1:,1]
        Y=np.transpose(Y)
        #########################################
        return X,Y



