#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 03:56:09 2023

@author: randon
"""


import random
from sympy import symbols
import copy
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

#operations = ["add","sub","mul","div","np.power","iden","np.sin","np.cos","np.tan",'np.log','diff',]

def create_expression(tree,unar,binar):
    #given a tree, it will parse its leaves to provide an expression in executable-python form. this can also be rendered by sympy to give a result. 
    #implicity assumes the algebra in question is associative. 
    """
    for a given node:
    def parse(node)
    if node not isleaf:
       parse(node.child)
    else:
        if node.parent is unary:
            node.parent.tag = parentfunc(nodedata)
            node.delete()
        else:
            if node.sibling isleaf:
                node.parent.tag = parentfunc(nodedata,node.siblinig.data)
            else:
                parse(node.sibling)
    """
    retstr = ""
    btree = copy.deepcopy(tree)
    def parse(tree,node,unar,binar):
        #print("parsing",node,node.successors(tree.identifier))
        if len(node.successors(tree.identifier))==0:
            #print("am leaf",node)
            if tree.get_node(node.predecessor(tree.identifier)) == None:
                #print("hmnn, I have no parent")
            #    #print(node)
                if node.tag in unar:
                    node.tag+="(x)"
                if node.tag in binar:
                    chose = ['x','k']
                    chose = chose[np.random.randint(0,1)]
                    node.tag+="(nixx,"+chose+")"
            else:
                if tree.get_node(node.predecessor(tree.identifier)).tag in unar:
                    #print("unar")
                    #print(tree.get_node(node.predecessor(tree.identifier)))
                    par = node.predecessor(tree.identifier)
                    tree.get_node(node.predecessor(tree.identifier)).tag +="("+node.tag+")"
                    #print(tree.get_node(node.predecessor(tree.identifier)))
                    tree.remove_node(node.identifier)
                    #print("1-node done")
                    parse(tree,tree.get_node(par),unar,binar)
                    
                elif tree.get_node(node.predecessor(tree.identifier)).tag in binar:
                    sibling = [x.identifier for x in tree.siblings(node.identifier)]
                    #print(sibling)
                    if tree.get_node(sibling[0]).is_leaf():
                        par = node.predecessor(tree.identifier)
                        #print(par)
                        tree.get_node(node.predecessor(tree.identifier)).tag = tree.get_node(node.predecessor(tree.identifier)).tag+"("+node.tag+","+tree.get_node(sibling[0]).tag+")"
                        tree.remove_node(node.identifier)
                        tree.remove_node(sibling[0])
                        #print("2-node done")
                        
                        parse(tree,tree.get_node(par),unar,binar)
                    else:
                        #print("recurse - sibling")
                        for sib in sibling:
                            parse(tree,tree.get_node(sib),unar,binar)
                else:
                    pass
                    #print("hem")
                    #print(node)
        
        else:
            #print("recurse!")
            for child in node.successors(tree.identifier):
                #print("found child")
                parse(tree,tree.get_node(child),unar,binar)
            #else:
            #    print(node)
            #    print("huh??")
    
    parse(btree,btree.get_node(tree.root),unar,binar)

    return btree.get_node(btree.root).tag

from treelib import Node, Tree
def makerandomtree(Nnodes):
    exprtree = Tree()
    numnodes = Nnodes
    exprtree.create_node('q',0)
    availparents = [str(i) for i in range(numnodes)]*2
    availparents = [int(i) for i in availparents]
    for i in range(numnodes-1):
        #gets list of available parents, chooses a random one, removes the chosen one from the list of avail parents, then creates a node with its parent as 
        #the chosen one
        par =  [j for j in availparents if j<(i+1)]
        
        chpar= par[random.randint(0,len(par)-1)]
        #print(par,chpar,availparents)
        #print("popping",chpar,"index",par.index(chpar))
        availparents.pop(availparents.index(chpar))
        #print("popped",availparents)
        #print(i+1,chpar,par,availparents)
        exprtree.create_node('q',i+1,parent=chpar)
    return exprtree
def decorate_tree(tree,types,unaries,binaries):
    unar = [x for x in unaries if x in types]
    #unar.append("iden")
    binar = [x for x in binaries if x in types]
    
    #unar = only the unaries, binar = binaries, sbinar = binaries,unaries,and special binaries_intchild
    #for a given tree with nodes, it will randomly decorate the nodes with the desired operations, according to its unary, or binary nature. It will also add the leaves
    #either k for const or x for var
    toadds1 = []
    toadds2=[]
    for node in tree.all_nodes_itr():
        #print(node,tree.children(node.identifier))
        if len(tree.children(node.identifier))==2:
            node.tag=binar[random.randint(0,len(binar)-1)]
        elif len(tree.children(node.identifier))==1:
            t = (binar+unar)[random.randint(0,len(binar+unar)-1)]
            if t in binar:
                toadds1.append(node.identifier)
            node.tag=t
        elif len(tree.children(node.identifier))==0:
            node.tag=(binar+unar)[random.randint(0,len((binar+unar))-1)]
            if node.tag in binar:
                toadds2.append(node.identifier)
            else:
                toadds1.append(node.identifier)
    for toadd1 in toadds1:
        t = 'x'
        tree.create_node(tag=t,parent=toadd1)
    for toadd2 in toadds2:
        t1 = 'x'
        chose =  int(np.round(random.randint(0,10)+4)/10)
        chose = 0 if chose==-1 else chose
        t2 = t = ['x','k'][chose]
        tree.create_node(tag=t1,parent=toadd2)
        tree.create_node(tag=t2,parent=toadd2)
    return tree

def create_random_expressions(numsets,types,unaries,binaries,maxops=None):
    types = np.array(types)
    """creates up to numsets functions of shape(2,100), where the 0th row is the X and the 1st row is the Y for functions, that include operations specified in types
    
    types would be things like [+,*, pow(), sin,cos,tan,ln,Bessel,]
    and then we will generate functions containing 1 or more of these elements
    
    the algorithm is as follows:
    
    a tree with number_of_operations nodes is randomly generated. Then the nodes are decorated with an appropriate element. Then the tree is parsed.
    
    ie lets say the operations are (+,* sin()), and number_of_operations is 4. +,* are binary operations, they take a left and right. sin() is a unary operation, it only takes a right. 
    
    A random tree is generated with 4 nodes.
    
    0 +
    1 +      *  
    2 sin()
    
    Since this tree has the dangling nodes sin() and *, the leaves must be respectively (x), and (x,x) for sin and *. Then on the 1st layer, 1 has 1 dangling node,  as it
    is a binary operation, so a leaf is added to it.
    Then the 0th layer is complete, and the overal graph is
    +
    + *
    sin(), x, x, x
    x
    
    Naturally the nodes in the root-way can require more children, but never less (ie the node with 2 children can't be unary)
    
    special treatment: pow ideally should be an integer as a child node, otherwise we just end up with x**x
    """
    
    #binaries = ["add","sub","mul","div","pow"]
    #unaries = ["diff","inte","sin",'tan',"log"]
    binaries_intchild = ['pow']
    x = symbols('x')
    trees = []
    expr = []
    for i in range(numsets):
        selected_elements = np.random.randint(0,len(types),size=random.randint(0,len(types)))
        selected_elements = types[selected_elements]
        print(selected_elements)
        if maxops==None:
            number_of_operations = random.randint(len(selected_elements),2*len(selected_elements))            
        else:
            number_of_operations=random.randint(1,maxops)
        print(number_of_operations)
        tree = makerandomtree(number_of_operations)
        #tree decoration - each node gets its operation type
        tree = decorate_tree(tree,types, unaries, binaries)
        trees.append(tree)
    for tree in trees:
        expr.append(create_expression(tree,unaries,binaries))
    return trees,expr
    
#trees,expressions = create_random_expressions(1,operations)
#arrays = []

import math
        
#trees[0].show()