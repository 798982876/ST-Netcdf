'''

This file contains the main program. The computation graph for ST-ResNet is built, launched in a session and trained here.
'''

import numpy as np
import netCDF4 as nc
from netCDF4 import Dataset
import tensorflow as tf
import json
from tqdm import tqdm
import math

from model_module.params import Params as param
from model_module.st_resnet import Graph
from model_module.utils import batch_generator
from predict_module.pai import caculte_pai_pei
import psycopg2
import os

'''
主函数 步骤
1 根据train_id获取数据(包括层级)
'''


def predict_execute(time_step,layerth):
    # build the computation graph
    root_path = "data/train_"+str(time_step)+"/"

    # 读取数据
    input_file_path =root_path+'input/'+str(layerth)+'/'+str(layerth)+'.nc' 
    dataset = Dataset(input_file_path)
    crimegroup = dataset['crime']
    crimegrid =crimegroup.variables['crimegrid']#获得变量
    data =crimegrid[:,:,:]#按维度取变量的值

    env_group = dataset['environment']
    environment =env_group.variables['environment']#获得变量
    data_out =environment[:,:,:]#按维度取变量的值

    model_path = root_path + 'model/ResNet' + str(param.num_of_residual_units) + '/' + str(layerth) + '/current.meta'
    model_path_restore = root_path + 'model/ResNet' + str(param.num_of_residual_units) + '/' + str(layerth) + '/current'
    output_path = root_path + 'output/'

    if not os.path.exists(output_path):
        os.makedirs(output_path)
   
    print("Computation graph for ST-ResNet loaded\n")

    #获得维度
    row = crimegroup['row'].size
    col = crimegroup['col'].size

    g = Graph(row,col)
   



    X = []
    Y = []
    Out = []
    # for d in range(168, len(data)):
    #     X.append([data[d - 1].tolist(), data[d - 24].tolist(), data[d - 168].tolist()])
    #     Y.append([data[d]])

    '''
    2*time_step h的
    '''
    for d in range(int(168/time_step), len(data) - 1, 2):
        X.append([(data[d - 1] + data[d - 2]).tolist(), (data[d - int(24/time_step)] + data[d - int(24/time_step)+1]).tolist(),
                  (data[d - int(168/time_step)] + data[d - int(168/time_step)+1]).tolist()])
        Y.append([data[d] + data[d + 1]])
        Out.append([data_out[d]])

    '''
    3h
    '''
    # for d in range(168, len(data) - 2, 3):
    #     X.append(
    #         [(data[d - 1] + data[d - 2] + data[d - 3]).tolist(), (data[d - 24] + data[d - 23] + data[d - 22]).tolist(),
    #          (data[d - 168] + data[d - 167] + data[d - 166]).tolist()])
    #     Y.append([data[d] + data[d + 1] + data[d + 2]])
    # X = []
    # for j in range(x_closeness.shape[0]):
    #     X.append([x_closeness[j].tolist(), x_period[j].tolist(), x_trend[j].tolist()])

    # create train-test split of data
    # train_index = int(round((0.8 * len(X)), 0))
    # train_index = 7956
    train_index =int(round((0.8 * len(X)), 0))
    train_end = len(X)
    xtest = X[train_index:]
    #print('xtest',xtest)
    ytest = Y[train_index:]
    
    outtest = Out[train_index:]
    # print(len(ytest))
    # xtest = X[train_index:]
    # ytest = Y[train_index:]

    xtest = np.array(xtest)
    ytest = np.array(ytest)
    outtest = np.array(outtest)
    print('ZJ:xtest for predict:',xtest.shape)
    
    # print(xtest[0][1])
    # obtain an interator for the next batch ze)
    # test_batch_generator = batch_generator(xtest, ytest, param.batch_size)
    test_batch_generator = batch_generator(xtest, ytest, outtest, param.batch_size)
    # print(test_batch_generator)
    real = []
    pred = []
    # indices=[]
    print("Start learning:")
    with tf.Session(graph=g.graph) as sess:
        sess.run(tf.global_variables_initializer())

        new_saver = tf.train.import_meta_graph(model_path)
        new_saver.restore(sess, model_path_restore)  # predicet
        num_batches = xtest.shape[0] // param.batch_size
        # num_batches = 21
        # print('range(num_batches):',range(num_batches))
        for b in tqdm(range(num_batches)):
            
            # x_batch, y_batch = next(test_batch_generator)
            # indices,x_batch, y_batch, out_batch = next(test_batch_generator)#返回索引
            x_batch, y_batch, out_batch= next(test_batch_generator)
            
            x_closeness = np.array(x_batch[:, 0].tolist())
            # print(x_closeness)
            x_period = np.array(x_batch[:, 1].tolist())
            x_trend = np.array(x_batch[:, 2].tolist())
            y_batch = np.array(y_batch[:, 0].tolist())
            out_batch = np.array(out_batch[:, 0].tolist())
            x_closeness = x_closeness[:, :, :, np.newaxis]
            x_period = x_period[:, :, :, np.newaxis]
            x_trend = x_trend[:, :, :, np.newaxis]
            #out_batch = out_batch[:, :, :, np.newaxis]
            outputs = sess.run(
                g.x_res,
                feed_dict={
                    g.c_inp: x_closeness,
                    g.p_inp: x_period,
                    g.t_inp: x_trend,
                    g.outside_condition: out_batch
                })
            # 降维
            outputs = np.squeeze(outputs)
            # print (outputs)
            outputs = outputs-0.1#大于0.1的为1
            for i in range(8):
                for i2 in range(len(outputs[1])):
                    for i3 in range(len(outputs[0][0])):
                        outputs[i][i2][i3] = math.ceil(outputs[i][i2][i3])
                        # outputs[i][i2][i3] = outputs[i][i2][i3]

            for i in range(8):
                
                real.append(y_batch[i])
                pred.append(outputs[i])
                
            # indices = indices[:num_batches*8]#依据utils生成器条件

    f = open(output_path + str(layerth) + '_real.npy', "wb")
  
    print('realpath:'+output_path + str(layerth) + '_real.npy')
    np.save(f, real)
    # print('ZJ:indices',max(indices),indices)
    # print('ZJ:real.shape:',real[0].shape,'real.length',len(real))
    
    f2 = open(output_path + str(layerth) + '_pred.npy', "wb")
    np.save(f2, pred)
    # print('ZJ:pred:',pred[0].shape)
    
    #将indices存入
    # f3 = open(output_path + str(layer) + '_indices.npy', "wb")
    # np.save(f3, indices)
    
    # 将最原始的ytest 存入
    # f4 = open(output_path + str(layer) + '_ytest.npy', "wb")
    # np.save(f4, ytest)
    
    
    # print('pred:',pred)
    # print("predict Done")


if __name__ == '__main__':
    time_step = 1
    for layerth in range(10, 12):
        print(layerth)
        predict_execute(time_step,layerth)
 
        # predict_path = 'data/train_'+str(time_step)+'/output/' + str(layerth) + '_pred.npy'
        # real_path = 'data/train_'+str(time_step)+'/output/' + str(layerth) + '_real.npy'
        # print(caculte_pai_pei(predict_path, real_path))