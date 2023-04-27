# -*- coding: utf-8 -*-
"""
@author: Syed Hasib Akhter Faruqui
@email: syed-hasib-akhter.faruqui@utsa.edu
@web: www.shafnehal.com

Note: Learning Rate Schedulers
"""

import tensorflow as tf

# Exponential Decay
def lr_scheduler_exponentil_decay(initial_learning_rate = 0.001, decay_steps = 100, decay_rate = 0.95):

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                                                                    initial_learning_rate,
                                                                    decay_steps=decay_steps,
                                                                    decay_rate=decay_rate,
                                                                    staircase=True
                                                                )
    return lr_schedule

# Inverse Time Decay
# InverseTimeDecay is suitable when you want the learning rate to decay slowly over time, especially when the model's improvement slows down.
def lr_scheduler_InverseTimeDecay(initial_learning_rate = 0.001, decay_steps = 100, decay_rate = 0.05):

    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
                                                                    initial_learning_rate,
                                                                    decay_steps=decay_steps,
                                                                    decay_rate=decay_rate,
                                                                    staircase=False
                                                                )
    return lr_schedule

# Piecewise Constant Decay
#  This is suitable when you want to manually specify the learning rate at different stages of training. This can be useful if you have specific knowledge about how your model should behave during training or if you want to fine-tune it.

def lr_scheduler_PiecewiseConstantDecay(boundaries = [500, 1000, 2000], values = [0.001, 0.0005, 0.0001, 0.00005]):

    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                                                                        boundaries=boundaries,
                                                                        values=values
                                                                    )
    return lr_schedule

# Polinomial Decay
# PolynomialDecay is suitable when you want the learning rate to decay smoothly over time following a polynomial function. The power parameter controls the shape of the polynomial function.
def lr_scheduler_PolinomialDecay(initial_learning_rate = 0.001, decay_steps = 100, end_learning_rate = 0.0001, power = 0.5):

    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                                                                initial_learning_rate,
                                                                decay_steps=decay_steps,
                                                                end_learning_rate=end_learning_rate,
                                                                power=power
                                                            )
    return lr_schedule

# Cosine Decay and Cosine Anealing sets
# Cosine Annealing schedules are suitable when you want the learning rate to follow a smooth cosine function. This can be helpful in avoiding local minima and promoting exploration during optimization. CosineDecayRestarts adds a restart mechanism to the schedule, allowing the learning rate to periodically reset and decay again, which can further improve optimization.

def lr_scheduler_CosineDecay(initial_learning_rate = 0.001, decay_steps = 100):
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                                                            initial_learning_rate,
                                                            decay_steps=decay_steps
                                                            )
    return lr_schedule

def lr_scheduler_CosineDecay_Restart(initial_learning_rate = 0.001, first_decay_steps = 100, t_mul = 2.0, m_mul = 1.0, alpha = 0.0):

    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
                                                                        initial_learning_rate,
                                                                        first_decay_steps,
                                                                        t_mul=t_mul,
                                                                        m_mul=m_mul,
                                                                        alpha=alpha
                                                                    )
    return lr_schedule