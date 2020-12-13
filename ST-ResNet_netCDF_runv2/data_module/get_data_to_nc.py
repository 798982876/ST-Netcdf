#从数据库中获得填充了时间、空间的数据
import psycopg2
import pandas as pd
import numpy as np
import netCDF4 as nc
from netCDF4 import Dataset
import os


#从数据库获得数据
def get_data_from_pg(start_time,end_time,time_step,space_extent,layerth):
    #查询语句，使得结果行列号从0开始，time_id,row,col,counts
    event_query_str = "select e.time_id,e.g_row-min_g_row as zero_row,e.g_col-min_g_col as zero_col,e.lat,e.lon,e.counts from \
        (select min(g_row) as min_g_row, min(g_col) as min_g_col from crimestat.f_get_resnet_data_1(%s,%s,%d,%s,%d)) min,\
        crimestat.f_get_resnet_data_1(%s,%s,%d,%s,%d) e order by e.time_id,zero_row,zero_col"%(start_time,end_time,time_step,space_extent,layerth,start_time,end_time,time_step,space_extent,layerth)
    
    
    # event_query_str = "select e.time_id,e.g_row,e.g_col,e.lat,e.lon,e.counts from crimestat.f_get_resnet_data(%s,%s,%d,%s,%d) e order by e.time_id,e.lat,e.lon"%(start_time,end_time,time_step,space_extent,layerth)
    
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
def prepare_data_to_nc(start_time,end_time,time_step,space_extent,layerth):
    data_from_pg = get_data_from_pg(start_time,end_time,time_step,space_extent,layerth)#格式：元组数组
    # data_from_csv = pd.read_csv('out.csv',header=None)#环境数据
    data_from_csv = np.loadtxt('data/out.csv',delimiter = ',')#环境数据
    
    data_arr = np.array(data_from_pg,dtype='float64')
    #获得最大最小行列号，及time_id
    max_time_id,min_time_id = max(data_arr[:,0]),min(data_arr[:,0])
    max_row,min_row = max(data_arr[:,1]),min(data_arr[:,1])
    max_col,min_col = max(data_arr[:,2]),min(data_arr[:,2])
    max_lat,min_lat = max(data_arr[:,3]),min(data_arr[:,3])
    max_lon,min_lon = max(data_arr[:,4]),min(data_arr[:,4])

    #组织数据
    dimension_time = int(max_time_id)-int(min_time_id)+1
    rows = int(max_row)-int(min_row)+1#对应lat
    cols = int(max_col)-int(min_col)+1#对应lon


    #环境数据获得，归一化后按时间空间展开
    #environment_value = change_out_np(data_from_csv,dimension_row,dimension_col)
    normal_env = (data_from_csv-np.min(data_from_csv))/(np.max(data_from_csv)-np.min(data_from_csv))
    #创建三维1矩阵
    matrix_env= np.ones((dimension_time, rows, cols))
    index = 0
    for i in normal_env:
        matrix_env[index:int(index+24/time_step),:,:] = i
        index+=int(24/time_step)
    env_value = matrix_env
 
    #犯罪数据的获得，按时空位置填充
      #创建三维0矩阵
    matrix_crime= np.zeros((dimension_time, rows, cols))
   #按index改数据
    for i in data_from_pg:
        index = (i[0]-1,i[1],i[2])
        count_value = i[5]
        matrix_crime.itemset(index,count_value)
    
   #返回变量值，包括维度变量和主变量
    times_value = np.arange(min_time_id,max_time_id+1,1)#前闭后开
    lats_value = np.arange(min_lat,max_lat+(max_lat-min_lat)/(rows-1),(max_lat-min_lat)/(rows-1))
    lons_value = np.arange(min_lon,max_lon+(max_lon-min_lon)/(cols-1),(max_lon-min_lon)/(cols-1))
    
    start_time_value = np.arange(min_time_id-1,max_time_id,1)
    end_time_value = np.arange(min_time_id,max_time_id+1,1)

    # lats_value = np.arange(min_lat,max_lat+1,(max_lat-min_lat)/(dimension_col-1))
    # lons_value = np.arange(min_lon,max_lon+1,(max_lon-min_lon)/(dimension_row-1))

    crimegrid_value = matrix_crime


    return rows,cols,times_value ,start_time_value,end_time_value,lats_value,lons_value, crimegrid_value,env_value
    

 #构建nc数据
def construct_nc(start_time,end_time,time_step,space_extent,layerth,data_path):

    #各变量数据准备
    rows,cols,times_value ,start_time_value,end_time_value,lats_value,lons_value, crimegrid_value,env_value = prepare_data_to_nc(start_time,end_time,time_step,space_extent,layerth)
    
    west = space_extent.split(',')[0][2:]
    south = space_extent.split(',')[1]
    east = space_extent.split(',')[2]
    north = space_extent.split(',')[3][:-2]
    
   


    #通过Dataset构造器 创建指定格式数据
    rootgroup = Dataset(data_path+str(layerth)+'.nc','w',format = 'NETCDF4')
    externalgroup = Dataset(data_path+str(layerth)+'_env.nc','w',format = 'NETCDF4')

    # 按层级划分数据，group包含 维度、属性、变量 和其他groups
    # crimegroup = rootgroup.createGroup('crime')
    # externalgroup = rootgroup.createGroup('environment')
  

    #创建维度，参数：名称、大小
    
    frame = rootgroup.createDimension('frame',None)
    lon = rootgroup.createDimension('lon',cols)
    lat = rootgroup.createDimension('lat',rows)
    
    frame_e = externalgroup.createDimension('frame',None)
    lon_e = externalgroup.createDimension('lon',cols)
    lat_e = externalgroup.createDimension('lat',rows)
    
    
    

    

    #创建变量，参数：名称、数据类型
    #维度可被定义为 坐标变量
    frames = rootgroup.createVariable('frame','u8',('frame',))
    lons = rootgroup.createVariable('lon','f8',('lon',))
    lats = rootgroup.createVariable('lat','f8',('lat',))
    
    frames_e = externalgroup.createVariable('frame','u8',('frame',))
    lons_e = externalgroup.createVariable('lon','f8',('lon',))
    lats_e = externalgroup.createVariable('lat','f8',('lat',))
    
    #存储每个time的起始时间和终止时间
    start_time_ = rootgroup.createVariable('start_time','f8',('frame',))
    end_time_ = rootgroup.createVariable('end_time','f8',('frame',))
    
    start_time_e = externalgroup.createVariable('start_time','f8',('frame',))
    end_time_e = externalgroup.createVariable('end_time','f8',('frame',))
    
    #创建变量
    crimegrid = rootgroup.createVariable('grid','f8',('frame','lat','lon'))
    environment = externalgroup.createVariable('environment','f8',('frame','lat','lon'))
    
    #设置属性，全局属性/变量属性
      #设置属性，全局属性/变量属性
    rootgroup.name = "ST-ResNet crime dataset"
    rootgroup.summary = ""
    rootgroup.description = "required crime data for ST-ResNet model"
    rootgroup.credits = ""
    rootgroup.extent = "extent:{'space':'west':%s,'south':%s,'east':%s,'north':%s},'time':{'start_time':%s,'end_time':%s,'interval':%d}"%(west,south,east,north,start_time,end_time,time_step)
    
    externalgroup.name = "ST-ResNet environment dataset"
    externalgroup.summary = ""
    externalgroup.description = "required environment data for ST-ResNet model"
    externalgroup.credits = ""
    externalgroup.extent = "extent:{'space':'west':%s,'south':%s,'east':%s,'north':%s},'time':{'start_time':%s,'end_time':%s,'interval':%d}"%(west,south,east,north,start_time,end_time,time_step)
    
    
    
    frames.units = 'time period number'
    lats.units = 'degrees_north'
    lons.units = 'degrees_east'
    
    frames_e.units = 'time period number'
    lats_e.units = 'degrees_north'
    lons_e.units = 'degrees_east'
   
    
    
    start_time_.units = 'hours since 2012-1-1 00:00:00'
    start_time_e.units = 'hours since 2012-1-1 00:00:00'
    
    end_time_.units = 'hours since 2012-1-1 00:00:00'
    end_time_e.units = 'hours since 2012-1-1 00:00:00'


    # 写入维度变量,连续的time_id,行列号
    frames[:] = times_value
    start_time_[:] = start_time_value
    end_time_[:] = end_time_value
    lats[:] = lats_value
    lons[:] = lons_value
    crimegrid[:,:,:] = crimegrid_value
    
    
    frames_e[:] = times_value
    start_time_e[:] = start_time_value
    end_time_e[:] = end_time_value
    lats_e[:] = lats_value
    lons_e[:] = lons_value
    environment[:,:,:] = env_value
    

   
    #写入变量，同时改变了无限维的长度

    # a = uniform(size = (180,3,3))
    # event[0:180,:,:] = a
    # #获得各维度范围，单独切片
    print(crimegrid.shape)
    print(environment.shape)
    piece = crimegrid[[1,2],0,[0,1,2]]
    print(piece)
    rootgroup.close()
    externalgroup.close()

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


