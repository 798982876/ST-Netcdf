'''

This file contains class Params for hyperparameter declarations.
'''
import numpy as np
# import data_module.data_get as data_get


class Params(object):
    batch_size = 8
    closeness_sequence_length = 1
    period_sequence_length = 1
    trend_sequence_length = 1
    nb_flow = 1
    num_of_filters = 64
    num_of_residual_units = 4
    num_of_output = 1  # depth of predicted output map
    delta = 0.5
    epsilon = 1e-7
    beta1 = 0.8
    beta2 = 0.999
    lr = 0.001
    num_epochs = 8

    def __init__(self, row,col):

        self.map_height = row
        self.map_width = col