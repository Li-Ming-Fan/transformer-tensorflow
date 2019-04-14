# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 21:24:16 2019

@author: limingfan
"""

from Zeras.layers import Dense
from Zeras.layers import build_module_copies

import tensorflow as tf


tf.reset_default_graph()


dc = build_module_copies(Dense, (10, 10), 3)
print(dc)
print(dc[0].wb)


#
a = [0,1,2,3,4]

data_iter = a.__iter__()

print(data_iter)

while True:
    print(next(data_iter))
    
    
    