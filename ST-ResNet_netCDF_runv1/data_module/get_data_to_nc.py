#从数据库中获得填充了时间、空间的数据
import psycopg2
import pandas as pd
import numpy as np
import netCDF4 as nc
from netCDF4 import Dataset
import os


#从数据库获得数据
def get_data_from_pg(start_time,end_time,time_step,space_extend,layerth):
    #查询语句，使得结果行列号从0开始，time_id,row,col,counts
    event_query_str = "select e.time_id,e.g_row-min_g_row as zero_row,e.g_col-min_g_col as zero_col,e.counts from \
        (select min(g_row) as min_g_row, min(g_col) as min_g_col from crimestat.f_get_resnet_data(%s,%s,%d,%s,%d)) min,\
        crimestat.f_get_resnet_data(%s,%s,%d,%s,%d) e order by e.time_id,zero_row,zero_col"%(start_time,end_time,time_step,space_extend,layerth,start_time,end_time,time_step,space_extend,layerth)
    conn = psycopg2.connect(
        database="crimeanalysis",
        user="crimeanalysis",
        password="589555",
        host="172.21.212.225",
        port="7032")
    cur = conn.cursor()
    cur.execute(event_query_str)
    rows = cur.fetchall()

    print('get data successfully from pg!')
    return rows

 


#数据填充
def prepare_data_to_nc(start_time,end_time,time_step,space_extend,layerth):
    data_from_pg = get_data_from_pg(start_time,end_time,time_step,space_extend,layerth)#格式：元组数组
    # data_from_csv = pd.read_csv('out.csv',header=None)#环境数据
    data_from_csv = np.loadtxt('data/out.csv',delimiter = ',')#环境数据
    
    data_arr = np.array(data_from_pg)
    #获得最大最小行列号，及time_id
    max_time_id,min_time_id = max(data_arr[:,0]),min(data_arr[:,0])
    max_row,min_row = max(data_arr[:,1]),min(data_arr[:,1])
    max_col,min_col = max(data_arr[:,2]),min(data_arr[:,2])

    #组织数据
    dimension_time = max_time_id-min_time_id+1
    dimension_row = max_row-min_row+1
    dimension_col = max_col-min_col+1

    #环境数据获得，归一化后按时间空间展开
    #environment_value = change_out_np(data_from_csv,dimension_row,dimension_col)
    normal_env = (data_from_csv-np.min(data_from_csv))/(np.max(data_from_csv)-np.min(data_from_csv))
    #创建三维1矩阵
    matrix_env= np.ones((dimension_time, dimension_row, dimension_col))
    index = 0
    for i in normal_env:
        matrix_env[index:int(index+24/time_step),:,:] = i
        index+=int(24/time_step)
    env_value = matrix_env
 
    #犯罪数据的获得，按时空位置填充
    #创建三维0矩阵
    matrix_crime= np.zeros((dimension_time, dimension_row, dimension_col))
   #按index改数据
    for i in data_from_pg:
        index = (i[0]-1,i[1],i[2])
        count_value = i[3]
        matrix_crime.itemset(index,count_value)
    
    #返回变量值，包括维度变量和主变量
    times_value = np.arange(min_time_id,max_time_id+1,1)#前闭后开
    rows_value = np.arange(min_row,max_row+1,1)
    cols_value = np.arange(min_col,max_col+1,1)

    crimegrid_value = matrix_crime


    return times_value ,rows_value,cols_value, crimegrid_value,env_value
    

 #构建nc数据
def construct_nc(start_time,end_time,time_step,space_extend,layerth,data_path):

    #各变量数据准备
    times_value ,rows_value,cols_value, crimegrid_value,env_value = prepare_data_to_nc(start_time,end_time,time_step,space_extend,layerth)


    #通过Dataset构造器 创建指定格式数据
    rootgroup = Dataset(data_path+str(layerth)+'.nc','w',format = 'NETCDF4')

    #按层级划分数据，group包含 维度、属性、变量 和其他groups
    crimegroup = rootgroup.createGroup('crime')
    externalgroup = rootgroup.createGroup('environment')
  

    #创建维度，参数：名称、大小
    crime_time = crimegroup.createDimension('time',None)
    crime_col = crimegroup.createDimension('col',None)
    crime_row = crimegroup.createDimension('row',None)

    env_time = externalgroup.createDimension('time',None)
    env_col = externalgroup.createDimension('col',None)
    env_row = externalgroup.createDimension('row',None)
    

    #创建变量，参数：名称、数据类型
    #维度可被定义为 坐标变量
    crime_times = crimegroup.createVariable('time','u8',('time',))
    crime_cols = crimegroup.createVariable('col','u8',('col',))
    crime_rows = crimegroup.createVariable('row','u8',('row',))

    env_times = externalgroup.createVariable('time','u8',('time',))
    env_cols = externalgroup.createVariable('col','u8',('col',))
    env_rows = externalgroup.createVariable('row','u8',('row',))
    
    #创建变量
    crimegrid = crimegroup.createVariable('crimegrid','f8',('time','row','col'))
    environment = externalgroup.createVariable('environment','f8',('time','row','col'))
    
    #设置属性，全局属性/变量属性
    crimegroup.description = "property crime group by grids and hours"
    externalgroup.description = "normal average temperature in a day"

    crimegroup.timeset = "from %s to %s  ;interval: %d"%(start_time,end_time,time_step)
    externalgroup.timeset = "from %s to %s  ;interval: %d"%(start_time,end_time,time_step)

    crime_times.units = '%d*hours since %s'%(time_step,start_time)
    crime_cols.units = '-km'
    crime_rows.units = '-km'

    env_times.units = '%d*hours since %s'%(time_step,start_time)
    env_cols.units = '-km'
    env_rows.units = '-km'
    crimegrid.units = 'pieces of crimes in a grid'

    #写入维度变量,连续的time_id,行列号
    crime_times[:] = times_value
    crime_rows[:] = rows_value
    crime_cols[:] = cols_value
    crimegrid[:,:,:] = crimegrid_value

    env_times[:] = times_value
    env_rows[:] = rows_value
    env_cols[:] = cols_value
    environment[:,:,:] = env_value
    

    #获得变量维度
    print(crimegrid.shape)
    print(environment.shape)
    piece = crimegrid[[1,2],0,[0,1,2]]

    #写入变量，同时改变了无限维的长度

    # a = uniform(size = (180,3,3))
    # event[0:180,:,:] = a
    # #获得各维度范围，单独切片
    rootgroup.close()

    print('yes')
    return 1




  




    











# a = [(1,2,3),(4,5,6),(7,8,9)]
# b = [list(c) for c in a]
# d = np.mat(a)
# e = d[:,0:1]
# f = np.array(a)
# h = f[:,1]

# i = np.arange(1, 10, 1)#前闭后开

#construct_nc("'2012-01-01 00:00:00'","'2013-01-01 00:00:00'",1,"'{119.914598,30.760336,121.382326,32.036541}'",10)

# print('yes')


