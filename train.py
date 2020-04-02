# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function

#  import dependency library
import sys
from shutil import copyfile
from tensorflow.contrib.layers.python.layers import regularizers

#  import user defined library
from Util.loss_function import *
from Util.data_loader import *
from Util.train_test_func import *
from Util.parse_config import parse_config
from Util.DMapNet import DMapNet


#  return network object
class NetFactory(object):
    @staticmethod
    def create(name):
        if name == 'DMapNet':
            return DMapNet
        # add your own networks here
        print('unsupported network:', name)
        exit()


#=============================================================
#          main function for training
#=============================================================
def train(config_file):


    #=============================================================
    #               1, Load configuration parameters
    #=============================================================
    config = parse_config(config_file)
    config_data = config['Data']
    config_net = config['network']
    config_train = config['training']
    random.seed(config_train.get('random_seed', 1))
    assert(config_data['with_ground_truth'])
    net_type = config_net['net_type']
    net_name = config_net['net_name']
    class_num = config_data['edt_discrete_num']
    label_edt_discrete = config_data['label_edt_discrete']
    batch_size = config_data.get('batch_size', 5)


    #==============================================================
    #               2, Construct computation graph
    #==============================================================
    full_data_shape = [batch_size] + config_data['data_shape']  # Batch size + original Data size
    tf.reset_default_graph()
    with tf.name_scope('model_builder'):
        x = tf.placeholder(tf.float32, shape = full_data_shape, name='Input')  # Place holder to transfer Data
        if label_edt_discrete:
            y = tf.placeholder(tf.uint8, [batch_size]+config_data['label_shape'], name='Label')
        else:
            y = tf.placeholder(tf.float32, [batch_size]+config_data['label_shape'], name='Label')
        w_regularizer = regularizers.l2_regularizer(config_train.get('decay', 1e-7))
        b_regularizer = regularizers.l2_regularizer(config_train.get('decay', 1e-7))
        net_class = NetFactory.create(net_type)
        net = net_class(num_classes = class_num,
                        w_regularizer = w_regularizer,
                        b_regularizer = b_regularizer,
                        name = net_name)
        net.set_params(config_net)
        predicty = net(x, is_training = True)
    with tf.name_scope("loss"):
        if label_edt_discrete:
            loss = weighted_one_hot_loss(predicty, y, class_num)
        else:
            loss = regression_loss(predicty, y)
        ##  ~~~~~inspect intermediate results~~~~(optional)
        # yshape = y.shape
        # gt = tf.squeeze(y)
        # sample_slice = gt[0, 0, :, :]
        # pred_out = tf.argmax(predicty, axis=-1)
        # pred_slice = pred_out[0, 0, :, :]
        # sample_slice = tf.cast(sample_slice, tf.float32) / 16
        # pred_slice = tf.cast(pred_slice, tf.float32) / 16
        # sample_slice = tf.reshape(sample_slice, [1, yshape[2], yshape[3], 1])
        # pred_slice = tf.reshape(pred_slice, [1, yshape[2], yshape[3], 1])
        # tf.summary.image('Input', sample_slice, 3)
        # tf.summary.image('Output', pred_slice, 3)
        # loss_summary = tf.summary.scalar('Loss', loss)
        #print('size of predicty:',predicty)

    #  initialize session and saver
    lr = config_train.get('learning_rate', 1e-3)
    opt_step = tf.train.AdamOptimizer(lr).minimize(loss)
    sess = tf.InteractiveSession()  # InteractiveSession sets itself as default session.
    sess.run(tf.global_variables_initializer())  # Give initial values to all variables.
    saver = tf.train.Saver()

    #  ##  ~~~~~inspect intermediate results~~~~(optional)
    # summ = tf.summary.merge_all()
    # merge_summary = tf.summary.merge([loss_summary])
    # if config_train.get('record_summary', False):
    #     summary_dir = config_train.get('summary_dir', None)
    #     assert summary_dir, 'Please specify the path for log saver'
    #     saver_summary = tf.summary.FileWriter(os.path.join(summary_dir, net_name))
    #     saver_summary.add_graph(sess.graph)


    #==============================================================
    #               3, Data loader
    #==============================================================
    dataloader = DataLoader(config_data)
    dataloader.load_data()


    # ==============================================================
    #               3, Start train
    # ==============================================================
    loss_file = config_train['model_save_prefix'] + "_loss.txt"
    start_it  = config_train.get('start_iteration', 0)
    if( start_it > 0):
        saver.restore(sess, config_train['model_pre_trained'])
    loss_list, temp_loss_list = [], []
    for n in range(start_it, config_train['maximal_iteration']):
        train_pair = dataloader.get_subimage_batch()
        input_x = train_pair['images']
        output_y = train_pair['labels']
        sess.run(opt_step, feed_dict={x:input_x, y:output_y})

        if(n%config_train['test_iteration'] == 0):
            batch_dice_list = []
            for step in range(config_train['test_step']):
                train_pair = dataloader.get_subimage_batch()
                input_x = train_pair['images']
                output_y = train_pair['labels']
                dice = loss.eval(feed_dict ={x:input_x, y:output_y})
                batch_dice_list.append(dice)
            batch_dice = np.asarray(batch_dice_list, np.float32).mean()
            t = time.strftime('%X %x %Z')
            print(t, 'iterration {:6d}: {:10.4f}'.format(n, batch_dice))
            #print(t, 'n', n,'loss', batch_dice)
            loss_list.append(batch_dice)
            np.savetxt(loss_file, np.asarray(loss_list))

        if((n+1)%config_train['snapshot_iteration']  == 0):
            saver.save(sess, config_train['model_save_prefix']+"_{0:}.ckpt".format(n+1))
        ##  ~~~~~inspect intermediate results~~~~(optional)
        # if(n%100 == 0):
        #     s = sess.run(summ, feed_dict={x: input_x, y: output_y})
        #     saver_summary.add_summary(s, n)
        # if(n%10==0):
        #     s = sess.run(loss_summary, feed_dict={x: input_x, y: output_y})
        #     saver_summary.add_summary(s, n)

    timestr = time.strftime('%m_%d_%H_%M')
    copyfile(config_file, os.path.join(os.path.dirname(config_train['model_save_prefix']), config_net['net_name']+'Paramters-'+timestr+'.txt'))
    sess.close()
    
if __name__ == '__main__':
    if(len(sys.argv) != 2):
        raise Exception("Invaid number of input parameters!")
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))  # make sure config_file is a file name
    train(config_file)
