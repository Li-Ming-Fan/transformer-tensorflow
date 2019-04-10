# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 21:24:16 2019

@author: limingfan
"""

from zoo_layers import Dense
from zoo_layers import build_module_copies

import tensorflow as tf


tf.reset_default_graph()


dc = build_module_copies(Dense, (10, 10), 3)
print(dc)
print(dc[0].wb)

