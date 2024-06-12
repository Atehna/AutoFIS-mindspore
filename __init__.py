# 设置全局的配置参数，比如数据路径，主机名，环境类型等等
import socket

config = {}
config['data_path'] = '/Avazu'  #add your path here

host = socket.gethostname()
config['host'] = host.lower()

config['env'] = 'gpu'
config['dtype'] = 'float32'
config['scale'] = 0.001
config['minval'] = - config['scale']
config['maxval'] = config['scale']
config['mean'] = 0
config['stddev'] = 0.001
config['sigma'] = config['stddev']
config['const_value'] = 0
config['rnd_type'] = 'uniform'
config['factor_type'] = 'avg'
config['magnitude'] = 3

