# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 12:13:35 2017

@author: romulo
"""

def EuclideanDistance(v1, v2):
    sum = 0.0
    for index in range(len(v1)):
        sum += (v1[index] - v2[index]) ** 2
    return sum ** 0.5