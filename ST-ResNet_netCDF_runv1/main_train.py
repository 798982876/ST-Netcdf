import data_module.get_data_to_nc as get_data_to_nc
# import data_module.data_converse as data_converse
import tensorflow as tf
from model_module.st_resnet import Graph
import numpy as np
from model_module.utils import batch_generator
from model_module.params import Params as param
from tqdm import tqdm
import netCDF4 as nc
from netCDF4 import Dataset
from data.grpc_st.dist import st_resnet_pb2, st_resnet_pb2_grpc
from concurrent import futures
import grpc
import time
import sys
import os

'''
1 根据train_id 定义根目录root_path
2 根据train_id 查询训练条件
3 根据train_id 到生产库中查询数据，并保存在root_path中
4 根据本地数据，train_id，以及layer 对数据进行四叉树编码，并填充转换，最终转换为numpy文件保存
在本地，放在input文件中

'''
_ONE_DAY_IN_SECONDS = 60 * 60 * 24


#获得多层数据
def train_get_data(start_time,end_time,time_step,space_extend):
    '''
    根据训练的train_id从生产数据库获取数据，并进行分析
    '''
    root_path = "data/train_"+str(time_step)+"/"
    # 获取多层的数据 并进行转换
    for layerth in range(10, 12):
        data_path = root_path+'input/'+str(layerth)+'/'
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        data_status = get_data_to_nc.construct_nc(start_time,end_time,time_step,space_extend,layerth,data_path)
        if data_status!=1:
            return 0
    print("Data processing has been completed!")
    return 1


def train_execute(time_step,layerth):
   
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

    #获得维度
    row = crimegroup['row'].size
    col = crimegroup['col'].size

    g = Graph(row,col)

    train_writer = tf.summary.FileWriter(
        root_path + 'logdir/train' + str(layerth), g.loss.graph)
    val_writer = tf.summary.FileWriter(root_path + 'logdir/val' + str(layerth),
                                       g.loss.graph)
    print("Computation graph for ST-ResNet loaded\n")
    # data = np.load(input_file_path)
    # data_out = np.load(input_out_file_path)




    X = []
    Y = []
    Out = []
    # 169是因为 一周168个小时,!!!除法有问题
    for d in range(0, len(data) - int(168/time_step) -1):
        X.append(
            [data[d].tolist(), data[d + int(24/time_step)].tolist(), data[d + int(168/time_step)].tolist()])
    for dy in range(int(168/time_step), len(data) - 1):
        Y.append([data[dy]])
    for d_out in range(len(data_out)):
        Out.append([data_out[d_out]])
    train_index = int(round((0.8 * len(X)), 0))
    xtrain = X[:train_index]
    ytrain = Y[:train_index]
    outtrain = Out[:train_index]
    xtest = X[train_index:]
    ytest = Y[train_index:]
    outtest = Out[train_index:]

    xtrain = np.array(xtrain)
    ytrain = np.array(ytrain)
    outtrain = np.array(outtrain)
    xtest = np.array(xtest)
    ytest = np.array(ytest)
    outtest = np.array(outtest)

    # obtain an interator for the next batch
    train_batch_generator = batch_generator(xtrain, ytrain, outtrain, param.batch_size)
    test_batch_generator = batch_generator(xtest, ytest, outtest, param.batch_size)
    print(param.batch_size)
    print("Start learning:")
    with tf.Session(graph=g.graph) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(param.num_epochs):
            loss_train = 0
            loss_val = 0
            print("Epoch: {}\t".format(epoch), )
            # training
            num_batches = xtrain.shape[0] // param.batch_size
            for b in tqdm(range(num_batches)):
                x_batch, y_batch, out_batch = next(train_batch_generator)
                x_closeness = np.array(x_batch[:, 0].tolist())
                x_period = np.array(x_batch[:, 1].tolist())
                x_trend = np.array(x_batch[:, 2].tolist())
                y_batch = np.array(y_batch[:, 0].tolist())
                out_batch = np.array(out_batch[:, 0].tolist())
                x_closeness = x_closeness[:, :, :, np.newaxis]
                x_period = x_period[:, :, :, np.newaxis]
                x_trend = x_trend[:, :, :, np.newaxis]
                y_batch = y_batch[:, :, :, np.newaxis]
                #out_batch = out_batch[:, :, :, np.newaxis]
                loss_tr, _, summary = sess.run(
                    [g.loss, g.optimizer, g.merged],
                    feed_dict={
                        g.c_inp: x_closeness,
                        g.p_inp: x_period,
                        g.t_inp: x_trend,
                        g.output: y_batch,
                        g.outside_condition: out_batch
                    })
                loss_train = loss_tr * param.delta + loss_train * (
                        1 - param.delta)
                train_writer.add_summary(summary, b + num_batches * epoch)

            # testing
            num_batches = xtest.shape[0] // param.batch_size
            for b in tqdm(range(num_batches)):
                x_batch, y_batch, out_batch = next(test_batch_generator)
                x_closeness = np.array(x_batch[:, 0].tolist())
                x_period = np.array(x_batch[:, 1].tolist())
                x_trend = np.array(x_batch[:, 2].tolist())
                y_batch = np.array(y_batch[:, 0].tolist())
                out_batch = np.array(out_batch[:, 0].tolist())

                x_closeness = x_closeness[:, :, :, np.newaxis]
                x_period = x_period[:, :, :, np.newaxis]
                x_trend = x_trend[:, :, :, np.newaxis]
                y_batch = y_batch[:, :, :, np.newaxis]
                # out_batch = out_batch[:, :, :, np.newaxis]

                loss_v, summary = sess.run(
                    [g.loss, g.merged],
                    feed_dict={
                        g.c_inp: x_closeness,
                        g.p_inp: x_period,
                        g.t_inp: x_trend,
                        g.output: y_batch,
                        g.outside_condition: out_batch
                    })
                loss_val += loss_v
                val_writer.add_summary(summary, b + num_batches * epoch)
            if (num_batches != 0):
                loss_val /= num_batches

            print("loss: {:.3f}, val_loss: {:.3f}".format(
                loss_train, loss_val))
            # save the model after every epoch
            g_path = root_path + 'model/ResNet' + str(param.num_of_residual_units) + '/' + str(layerth)
            if not os.path.exists(g_path):
                os.makedirs(g_path)
            g.saver.save(sess, g_path + "/current")
    train_writer.close()
    val_writer.close()
    # insert_loss_str = "update task.train set train_condition = jsonb_insert(train_condition,\'{layer_info,layer" + str(
    #     layer) + ",loss}\',\'" + str(loss_train) + "\')where train_id = \'" + str(
    #     train_id) + "\' returning train_condition;"
    # data_get.operate_task(insert_loss_str)
    return 1





if __name__ == '__main__':
    time_step = 1
    train_get_data("'2012-01-01 00:00:00'","'2013-01-01 00:00:00'",time_step,"'{119.914598,30.760336,121.382326,32.036541}'")
    for i in range(10, 12):
        print('layer:',i)      
        train_execute(time_step,i)
      
