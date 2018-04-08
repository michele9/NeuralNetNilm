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
import NeuralNetNilm.nilm_metric as nm
from matplotlib.pylab import *

def remove_space(string):
    return string.replace(" ","")
    
def get_arguments():
    parser = argparse.ArgumentParser(description='Predict the appliance\
                                     give a trained neural network\
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
                        default='./data/uk-dale/testdata/',
                        help='this is the directory of the training samples')
    parser.add_argument('--batchsize',
                        type=int,
                        default=1000,
                        help='The batch size of training examples')
    parser.add_argument('--n_epoch',
                        type=int,
                        default=1,
                        help='The number of epoches.')
    parser.add_argument('--nosOfWindows',
                        type=int,
                        default=100,
                        help='The number of windows for prediction \
                        for each iteration.')
    parser.add_argument('--save_model',
                        type=int,
                        default=-1,
                        help='Save the learnt model: \
                            0 -- not to save the learnt model parameters;\
                            n (n>0) -- to save the model params every n steps;\
                            -1 -- only save the learnt model params \
                                    at the end of traing.')
    return parser.parse_args()

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
def load_dataset():
    app = args.datadir + args.appliance_name +'/' +'building2_'+ args.appliance_name
    test_set_x = np.load(app+'_test_x.npy')  
    test_set_y = np.load(app+'_test_y.npy')  
    ground_truth = np.load(app+'_test_gt.npy')  
    print('test set:', test_set_x.shape, test_set_y.shape)
    print('testset path:{}'.format(app+'_test_gt.npy'))
    print('testset path:{}'.format(app+'_test_x.npy'))
    print('testset path:{}'.format(app+'_test_y.npy'))
    
    return test_set_x, test_set_y, ground_truth

test_set_x, test_set_y, ground_truth = load_dataset()

shuffle = False
appliance_name = args.appliance_name
mean = params_appliance[appliance_name]['mean']
std = params_appliance[appliance_name]['std']
sess = tf.InteractiveSession()


windowlength = params_appliance[appliance_name]['windowlength']

offset = int(0.5*(params_appliance[appliance_name]['windowlength']-1.0))

test_kwag = {
    'inputs':test_set_x, 
    'targets':  ground_truth,
    'flatten':False}

# val_kwag = {
#     'inputs': val_set_x, 
#     'targets': val_set_y,
#     'flatten':False}

test_provider = NeuralNetNilm.DataProvider.MultiApp_Slider(batchsize = batchsize,
                                                 shuffle = False, offset=offset)
# val_provider = DataProvider.DoubleSourceSlider(batchsize = 5000, 
#                                                  shuffle = False, offset=offset)


x = tf.placeholder(tf.float32, 
                   shape=[None, windowlength],
                   name='x')
y_ = tf.placeholder(tf.float32, shape=[None, 1], name='y_')


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


param_file = 'cnn_s2s_' + args.appliance_name + '_pointnet_model'
model.load_weights(param_file + '_weights.h5')
print('params done')

test_prediction = nf.custompredict_add(sess=sess,
                                       network=model,
                                       output_provider = test_provider ,
                                       x = x,
                                       fragment_size=windowlength,
                                       output_length=windowlength,
                                       y_op=None,
                                       out_kwag=test_kwag,
                                       seqlength = test_set_x.size,
                                       std=std,
                                       mean=mean)

max_power = params_appliance[appliance_name]['max_on_power']
threshold = params_appliance[appliance_name]['on_power_threshold']


ground_truth = ground_truth[offset:-offset]*std+mean
mains = (test_set_x[offset:-offset])*std+mean

prediction = test_prediction[offset:-offset]
prediction[prediction<=0.0] = 0.0
print(prediction.shape)
print(ground_truth.shape)
# np.save(args.appliance_name.replace(" ","_")+'_prediction', prediction)
# to load results: np.load(args.appliance_name+'_prediction')
sess.close()
sample_second = 6.0 # sample time is 6 seconds
print('F1:{0}'.format(nm.get_F1(ground_truth.flatten(), prediction.flatten(), threshold)))
print('NDE:{0}'.format(nm.get_nde(ground_truth.flatten(), prediction.flatten())))
print('MAE:{0}'.format(nm.get_abs_error(ground_truth.flatten(), prediction.flatten())))
print('SAE:{0}'.format(nm.get_sae(ground_truth.flatten(), prediction.flatten(), sample_second)))
save_name_y_pred = 'results/'+'pointnet_building2_'+args.appliance_name+'_pred.npy' #save path for mains
save_name_y_gt = 'results/'+'pointnet_building2_'+args.appliance_name+'_gt.npy'#save path for target
save_name_mains = 'results/'+'pointnet_building2_'+args.appliance_name+'_mains.npy'#save path for target
np.save(save_name_y_pred, prediction)
np.save(save_name_y_gt,ground_truth)
np.save(save_name_mains,mains)

