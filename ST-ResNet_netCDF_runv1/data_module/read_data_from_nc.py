#获得写入的nc数据
import numpy as np
import netCDF4 as nc
from netCDF4 import Dataset


def read_nc(file):
    dataset = Dataset(file)
    crimegroup = dataset['crime']
    crimegrid =crimegroup.variables['crimegrid']#获得变量
    row =  crimegroup['row'].size
    crimedata =crimegrid[:,:,:]#按维度取变量的值

    env_group = dataset['environment']
    environment =env_group.variables['environment']#获得变量
    env_data =environment[:,:,:]#按维度取变量的值



    print('yes')

read_nc('data/st-resnet.nc')



