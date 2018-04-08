############################################################
# This code is to train a neural network to perform energy disaggregation, 
# i.e., given a sequence of electricity mains reading, the algorithm
# separates the mains into appliances.
#
# Inputs: mains windows -- find the window length in params_appliance
# Targets: appliances windows -- 
#
#
# This code is written by Chaoyun Zhang and Mingjun Zhong.
# Reference:
# Chaoyun Zhang, Mingjun Zhong, Zongzuo Wang, Nigel Goddard, and Charles Sutton.
# ``Sequence-to-point learning with neural networks for nonintrusive load monitoring." 
# Thirty-Second AAAI Conference on Articial Intelligence (AAAI-18), Feb. 2-7, 2018.
############################################################

import NeuralNetNilm.NetFlowExt as nf
import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Reshape
import numpy as np
import NeuralNetNilm.DataProvider
import argparse

# only one GPU is visible to current task.
CUDA_VISIBLE_DEVICES=0 

def remove_space(string):
    return string.replace(" ","")
    
def get_arguments():
    parser = argparse.ArgumentParser(description='Train a neural network\
                                     for energy disaggregation - \
                                     network input = mains window; \
                                     network target = the states of \
                                     the target appliance.')
    parser.add_argument('--appliance_name', 
                        type=remove_space,
                        default='kettle',
                        help='the name of target appliance')
    parser.add_argument('--datadir',
                        type=str,
                        default='./data/uk-dale/trainingdata/small/',
                        help='this is the directory of the training samples')
    parser.add_argument('--batchsize',
                        type=int,
                        default=1000,
                        help='The batch size of training examples')
    parser.add_argument('--n_epoch',
                        type=int,
                        default=50,
                        help='The number of epoches.')
    parser.add_argument('--save_model',
                        type=int,
                        default=-1,
                        help='Save the learnt model: \
                            0 -- not to save the learnt model parameters;\
                            n (n>0) -- to save the model params every n steps;\
                            -1 -- only save the learnt model params \
                                    at the end of training.')
    return parser.parse_args()

# Units:
# windowlength: number of data points
# on_power_threshold,max_on_power: power
#params_appliance = {'kettle':{'windowlength':129, 
#                              'on_power_threshold':2000, 
#                              'max_on_power':3948},
#                    'microwave':{'windowlength':129,
#                              'on_power_threshold':200,
#                              'max_on_power':3138},
#                    'fridge':{'windowlength':299,
#                              'on_power_threshold':50,
#                              'max_on_power':2572},
#                    'dishwasher':{'windowlength':599,
#                              'on_power_threshold':10,
#                              'max_on_power':3230},
#                    'washingmachine':{'windowlength':599,
#                              'on_power_threshold':20,
#                              'max_on_power':3962}}
                              
params_appliance = {'kettle':{'windowlength':599,
                              'on_power_threshold':2000,
                              'max_on_power':3998,
                             'mean':700,
                             'std':1000,
                             's2s_length':128},
                    'microwave':{'windowlength':599,
                              'on_power_threshold':200,
                              'max_on_power':3969,
                                'mean':500,
                                'std':800,
                                's2s_length':128},
                    'fridge':{'windowlength':599,
                              'on_power_threshold':50,
                              'max_on_power':3323,
                             'mean':200,
                             'std':400,
                             's2s_length':512},
                    'dishwasher':{'windowlength':599,
                              'on_power_threshold':10,
                              'max_on_power':3964,
                                  'mean':700,
                                  'std':1000,
                                  's2s_length':1536},
                    'washingmachine':{'windowlength':599,
                              'on_power_threshold':20,
                              'max_on_power':3999,
                                      'mean':400,
                                      'std':700,
                                      's2s_length':2000}}
                                      
args = get_arguments()
print(args.appliance_name)
appliance_name = args.appliance_name
def load_dataset():   
    tra_x = args.datadir+args.appliance_name+'_mains_'+'tra_small' #save path for mains
    val_x = args.datadir+args.appliance_name+'_mains_'+'val'

    tra_y = args.datadir+args.appliance_name+'_'+'tra_small'+'_'+'pointnet'#save path for target
    val_y = args.datadir+args.appliance_name+'_'+'val'+'_'+'pointnet'
    
    tra_set_x = np.load(tra_x+'.npy')  
    tra_set_y = np.load(tra_y+'.npy')  
    val_set_x = np.load(val_x+'.npy')  
    val_set_y = np.load(val_y+'.npy')  
    
    print('training set:', tra_set_x.shape, tra_set_y.shape)
    print('validation set:', val_set_x.shape, val_set_y.shape)
    
    return tra_set_x, tra_set_y, val_set_x,  val_set_y

# load the data set
tra_set_x, tra_set_y, val_set_x,  val_set_y = load_dataset()

# get the window length of the training examples
windowlength = params_appliance[args.appliance_name]['windowlength']

sess = tf.InteractiveSession()


offset = int(0.5*(params_appliance[appliance_name]['windowlength']-1.0))

tra_kwag = {
    'inputs': tra_set_x,
    'targets': tra_set_y,
    'flatten':False}

val_kwag = {
    'inputs': val_set_x,
    'targets': val_set_y,
    'flatten':False}

# tra_provider = DataProvider.DoubleSourceSlider(batchsize = batchsize,
#                                                  shuffle = True, offset=offset)
# val_provider = DataProvider.DoubleSourceSlider(batchsize = 5000,
#                                                  shuffle = False, offset=offset)

tra_provider = NeuralNetNilm.DataProvider.S2S_Slider(batchsize = args.batchsize,
                                                 shuffle = True, length = params_appliance[appliance_name]['windowlength'])
val_provider = NeuralNetNilm.DataProvider.S2S_Slider(batchsize = 5000,
                                                 shuffle = False, length = params_appliance[appliance_name]['windowlength'])


x = tf.placeholder(tf.float32,
                   shape=[None, windowlength],
                   name='x')
y_ = tf.placeholder(tf.float32, shape=[None, windowlength], name='y_')

# Network
inp = Input(tensor=x)

reshape = Reshape((-1, 599, 1),
                  )(inp)

cnn1 = Conv2D(30, kernel_size=(10, 1),
              strides=(1, 1),
              padding='same',
              activation='relu',
              )(reshape)

cnn2 = Conv2D(30, kernel_size=(8, 1),
              strides=(1, 1),
              padding='same',
              activation='relu',
              )(cnn1)

cnn3 = Conv2D(40, kernel_size=(6, 1),
              strides=(1, 1),
              padding='same',
              activation='relu',
              )(cnn2)

cnn4 = Conv2D(50, kernel_size=(5, 1),
              strides=(1, 1),
              padding='same',
              activation='relu',
              )(cnn3)

flat = Flatten()(cnn4)

d = Dense(1024, activation='relu')(flat)

d_out = Dense(windowlength, activation='relu')(d)

model = Model(inputs=inp, outputs=d_out)

print(model.summary())

y = model.outputs

cost = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(y, y_), 1))
#train_params = network.all_params
train_op = tf.train.AdamOptimizer(learning_rate=0.001,
                                  beta1=0.9,
                                  beta2=0.999,
                                  epsilon=1e-08,
                                  use_locking=False).minimize(cost,
                                                              #var_list=train_params
                                                              )

# initialize all variables
sess.run(tf.global_variables_initializer())
# params = tl.files.load_npz(path='', name='cnn_lstm_model.npz')
# tl.files.assign_params(sess, params, network)
print('set sucessful')

save_path = './cnn_s2s_' + appliance_name + '_pointnet_model'


nf.customfit(sess = sess,
             network = model,
             cost = cost,
             train_op = train_op,
             tra_provider = tra_provider,
             x = x,
             y_ = y_,
             acc=None,
             n_epoch= args.n_epoch,
             print_freq=1,
             val_provider=val_provider,
             save_model=args.save_model,
             tra_kwag=tra_kwag,
             val_kwag=val_kwag ,
             save_path=save_path,
             epoch_identifier=None,
             earlystopping=True,
             min_epoch=1,
             patience=10)

sess.close()
