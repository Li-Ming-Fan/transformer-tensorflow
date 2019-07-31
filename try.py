# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 21:24:16 2019

@author: limingfan
"""



import tensorflow as tf

tf.reset_default_graph()

ad = [ 8.32329655,  8.32329655,  8.32329845,  8.32329845,  8.32329845,  8.32329845,
  8.32329845,  8.32329941,  8.28163242,  8.32329845,  8.32329845,  8.32329655,
  8.32329941, 8.32329655,  8.32329845 , 8.32329655 , 8.32329845 , 8.32329845,
  8.32329655,  8.3231554,   8.32329655,  8.32329845,  8.32325554,  8.32329845,
  8.32329655,  8.28935242,  8.32329655, 8.3232975  , 8.32329941 , 8.32329845,
  8.32329845,  8.32329082,  8.32329845,  8.32329655,  8.32329845,  8.32329845,
  8.32329845,  8.32329845,  8.32329655,  8.3232975 ,  8.32329655,  8.32317448,
  8.32329655,  8.3231802 ,  8.32329845,  8.27982044,  8.32329655,  8.32329845,
  8.32329845,  8.32329845,  8.3232975 ,  8.32329845,  8.3232975 ,  8.3232975,
  8.32329845,  8.3232975 ,  8.27784348,  8.32329845,  8.32308769, 8.32329845,
  8.32314777,  8.30504799,  8.32329845, 8.3232975 ]
  
a = tf.Variable(ad)
b = tf.get_variable('b', shape = (2,3,5), initializer = tf.random_normal_initializer() )


batch_start = tf.cast(tf.ones_like(b[:, 0:1, 0:1]), dtype = tf.int64)
batch_start = tf.multiply(batch_start, 2)
batch_start = tf.squeeze(batch_start, axis = [-1])

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(a))
print(sess.run(b))
print()
print(sess.run(batch_start))
print()

c = tf.reduce_mean(a)

print(sess.run(c))

print()
print(sess.run(tf.cast([0, 1.2, -1.0], dtype = tf.bool)) )


