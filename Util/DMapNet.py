from __future__ import absolute_import, print_function

# import dependency library
import tensorflow as tf
from niftynet.layer.bn import BNLayer
from niftynet.layer.activation import ActiLayer
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.elementwise import ElementwiseLayer
from niftynet.layer.deconvolution import DeconvolutionalLayer
from niftynet.layer.convolution import ConvLayer, ConvolutionalLayer


class DMapNet(TrainableLayer):
    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='prelu',
                 name='DMapNet'):

        super(DMapNet, self).__init__(name=name)
        self.num_classes = num_classes
        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}
        self.acti_func = acti_func
        self.base_chns = [16, 32, 64, 64]

    def set_params(self, params):
        self.base_chns = params.get('base_feature_number', [32, 32, 64, 64])
        self.acti_func = params.get('acti_func', 'prelu')
        self.print_summary = params.get('print_summary', False)

    def layer_op(self, images, is_training):
        block1_1 = ResBlock(self.base_chns[0],
                  kernels = [[1, 3, 3], [1, 3, 3]],
                  acti_func=self.acti_func,
                  w_initializer=self.initializers['w'],
                  w_regularizer=self.regularizers['w'],
                  name = 'block1_1')

        block1_2 = ResBlock(self.base_chns[0],
                  kernels = [[1, 3, 3], [1, 3, 3]],
                  acti_func=self.acti_func,
                  w_initializer=self.initializers['w'],
                  w_regularizer=self.regularizers['w'],
                  name = 'block1_2')


        block2_1 = ResBlock(self.base_chns[1],
                  kernels = [[1, 3, 3], [1, 3, 3]],
                  acti_func=self.acti_func,
                  w_initializer=self.initializers['w'],
                  w_regularizer=self.regularizers['w'],
                  name = 'block2_1')

        block2_2 = ResBlock(self.base_chns[1],
                  kernels = [[1, 3, 3], [1, 3, 3]],
                  acti_func=self.acti_func,
                  w_initializer=self.initializers['w'],
                  w_regularizer=self.regularizers['w'],
                  name = 'block2_2')

        block3_1 =  ResBlock(self.base_chns[2],
                  kernels = [[1, 3, 3], [1, 3, 3]],
                  dilation_rates = [[1, 1, 1], [1, 1, 1]],
                  acti_func=self.acti_func,
                  w_initializer=self.initializers['w'],
                  w_regularizer=self.regularizers['w'],
                  name = 'block3_1')

        block3_2 =  ResBlock(self.base_chns[2],
                  kernels = [[1, 3, 3], [1, 3, 3]],
                  dilation_rates = [[1, 2, 2], [1, 2, 2]],
                  acti_func=self.acti_func,
                  w_initializer=self.initializers['w'],
                  w_regularizer=self.regularizers['w'],
                  name = 'block3_2')

        block4_1 =  ResBlock(self.base_chns[3],
                  kernels = [[1, 3, 3], [1, 3, 3]],
                  dilation_rates = [[1, 3, 3], [1, 3, 3]],
                  acti_func=self.acti_func,
                  w_initializer=self.initializers['w'],
                  w_regularizer=self.regularizers['w'],
                  name = 'block4_1')

        block4_2 =  ResBlock(self.base_chns[3],
                  kernels = [[1, 3, 3], [1, 3, 3]],
                  dilation_rates = [[1, 2, 2], [1, 2, 2]],
                  acti_func=self.acti_func,
                  w_initializer=self.initializers['w'],
                  w_regularizer=self.regularizers['w'],
                  name = 'block4_2')

        fuse1 = ConvolutionalLayer(self.base_chns[0],
                    kernel_size= [3, 1, 1],  # Convolution on intra layers
                    padding='VALID',
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    b_initializer=self.initializers['b'],
                    b_regularizer=self.regularizers['b'],
                    acti_func=self.acti_func,
                    name='fuse1')

        downsample1 = ConvolutionalLayer(self.base_chns[0],
                    kernel_size= [1, 3, 3],
                    stride = [1, 2, 2],
                    padding='SAME',
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    b_initializer=self.initializers['b'],
                    b_regularizer=self.regularizers['b'],
                    acti_func=self.acti_func,
                    name='downsample1')

        fuse2 = ConvolutionalLayer(self.base_chns[1],
                    kernel_size= [3, 1, 1],
                    padding='VALID',
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    b_initializer=self.initializers['b'],
                    b_regularizer=self.regularizers['b'],
                    acti_func=self.acti_func,
                    name='fuse2')

        downsample2 = ConvolutionalLayer(self.base_chns[1],
                    kernel_size= [1, 3, 3],
                    stride = [1, 2, 2],
                    padding='SAME',
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    b_initializer=self.initializers['b'],
                    b_regularizer=self.regularizers['b'],
                    acti_func=self.acti_func,
                    name='downsample2')

        fuse3 = ConvolutionalLayer(self.base_chns[2],
                    kernel_size= [3, 1, 1],
                    padding='VALID',
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    b_initializer=self.initializers['b'],
                    b_regularizer=self.regularizers['b'],
                    acti_func=self.acti_func,
                    name='fuse3')

        fuse4 = ConvolutionalLayer(self.base_chns[3],
                    kernel_size= [3, 1, 1],
                    padding='VALID',
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    b_initializer=self.initializers['b'],
                    b_regularizer=self.regularizers['b'],
                    acti_func=self.acti_func,
                    name='fuse4')

        feature_expand1 =  ConvolutionalLayer(self.base_chns[1],  # Output channels
                    kernel_size= [1, 1, 1],
                    stride = [1, 1, 1],
                    padding='SAME',
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    b_initializer=self.initializers['b'],
                    b_regularizer=self.regularizers['b'],
                    acti_func=self.acti_func,
                    name='feature_expand1')

        feature_expand2 =  ConvolutionalLayer(self.base_chns[2],
                    kernel_size= [1, 1, 1],
                    stride = [1, 1, 1],
                    padding='SAME',
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    b_initializer=self.initializers['b'],
                    b_regularizer=self.regularizers['b'],
                    acti_func=self.acti_func,
                    name='feature_expand2')

        feature_expand3 =  ConvolutionalLayer(self.base_chns[3],
                    kernel_size= [1, 1, 1],
                    stride = [1, 1, 1],
                    padding='SAME',
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    b_initializer=self.initializers['b'],
                    b_regularizer=self.regularizers['b'],
                    acti_func=self.acti_func,
                    name='feature_expand3')

        centra_slice1 = TensorSliceLayer(margin = 2)
        centra_slice2 = TensorSliceLayer(margin = 1)

        image_resize1 = ImageResize()
        image_resize2 = ImageResize()

        pred_up1  = DeconvolutionalLayer(self.num_classes,
                    kernel_size= [1, 3, 3],
                    stride = [1, 2, 2],
                    padding='SAME',
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    b_initializer=self.initializers['b'],
                    b_regularizer=self.regularizers['b'],
                    acti_func=self.acti_func,
                    name='pred_up1')
        pred_up2_1  = DeconvolutionalLayer(self.num_classes*2,
                    kernel_size= [1, 3, 3],
                    stride = [1, 2, 2],
                    padding='SAME',
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    b_initializer=self.initializers['b'],
                    b_regularizer=self.regularizers['b'],
                    acti_func=self.acti_func,
                    name='pred_up2_1')
        pred_up2_2  = DeconvolutionalLayer(self.num_classes*2,
                    kernel_size= [1, 3, 3],
                    stride = [1, 2, 2],
                    padding='SAME',
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    b_initializer=self.initializers['b'],
                    b_regularizer=self.regularizers['b'],
                    acti_func=self.acti_func,
                    name='pred_up2_2')
        pred_up3_1  = DeconvolutionalLayer(self.num_classes*4,
                    kernel_size= [1, 3, 3],
                    stride = [1, 2, 2],
                    padding='SAME',
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    b_initializer=self.initializers['b'],
                    b_regularizer=self.regularizers['b'],
                    acti_func=self.acti_func,
                    name='pred_up3_1')
        pred_up3_2  = DeconvolutionalLayer(self.num_classes*4,
                    kernel_size= [1, 3, 3],
                    stride = [1, 2, 2],
                    padding='SAME',
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    b_initializer=self.initializers['b'],
                    b_regularizer=self.regularizers['b'],
                    acti_func=self.acti_func,
                    name='pred_up3_2')

        final_pred =  ConvLayer(self.num_classes, # Output two class: target and background
                kernel_size=[1, 3, 3],
                padding = 'SAME',  # Same: keep shape; Valid: only get pixels with valid calculation.
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
                b_initializer=self.initializers['b'],
                b_regularizer=self.regularizers['b'],
                name='final_pred')

        f1 = images
        f1 = block1_1(f1, is_training)
        f1 = block1_2(f1, is_training)
        f1 = fuse1(f1, is_training)
        f1 = downsample1(f1, is_training)

        img_resize1 = image_resize1(images, f1)
        img_resize1 = centra_slice2(img_resize1)
        f1 = tf.concat([img_resize1, f1], axis=4, name='concate')
        if(self.base_chns[0] != self.base_chns[1]):
            f1 = feature_expand1(f1, is_training)  # To keep same channel number in cascaded netblocks
        f1 = block2_1(f1, is_training)
        f1 = block2_2(f1, is_training)
        f1 = fuse2(f1, is_training)

        f2 = downsample2(f1, is_training)
        img_resize2 = image_resize2(images, f2)
        img_resize2 = centra_slice1(img_resize2)
        f2 = tf.concat([img_resize2, f2], axis=4, name='concate')
        if(self.base_chns[1] != self.base_chns[2]):
            f2 = feature_expand2(f2, is_training)
        f2 = block3_1(f2, is_training)
        f2 = block3_2(f2, is_training)
        f2 = fuse3(f2, is_training)

        f3 = f2
        if(self.base_chns[2] != self.base_chns[3]):
            f3 = feature_expand3(f3, is_training)
        f3 = block4_1(f3, is_training)
        f3 = block4_2(f3, is_training)
        f3 = fuse4(f3, is_training)

        p1 = centra_slice1(f1)
        p1 = pred_up1(p1, is_training)

        p2 = centra_slice2(f2)
        p2 = pred_up2_1(p2, is_training)
        p2 = pred_up2_2(p2, is_training)

        p3 = pred_up3_1(f3, is_training)
        p3 = pred_up3_2(p3, is_training)

        cat = tf.concat([p1, p2, p3], axis=4, name='concate')
        pred = final_pred(cat)

        return pred

class ResBlock(TrainableLayer):
    def __init__(self,
                 n_output_chns,
                 kernels=[[1, 3, 3], [1, 3, 3]],
                 strides=[[1, 1, 1], [1, 1, 1]],
                 dilation_rates = [[1, 1, 1], [1, 1, 1]],
                 acti_func='prelu',
                 w_initializer=None,
                 w_regularizer=None,
                 with_res=True,
                 name='ResBlock'):
        super(ResBlock, self).__init__(name=name)  # Embed the network with the same name as parent
        self.n_output_chns = n_output_chns  # The number of channel in the output
        if hasattr(kernels, "__iter__"):  # a list of layer kernel_sizes
            assert(len(kernels) == len(strides))
            assert(len(kernels) == len(dilation_rates))
            self.kernels = kernels
            self.strides = strides
            self.dilation_rates = dilation_rates
        else:  # is a single number (indicating single layer)
            self.kernels = [kernels]
            self.strides = [strides]
            self.dilation_rates = [dilation_rates]
        self.acti_func = acti_func
        self.with_res = with_res

        self.initializers = {'w': w_initializer}
        self.regularizers = {'w': w_regularizer}

    def layer_op(self, input_tensor, is_training):
        output_tensor = input_tensor
        for i in range(len(self.kernels)):
            # create parameterised layers
            bn_op = BNLayer(regularizer=self.regularizers['w'],  # Add regulizer for samplicity
                            name='bn_{}'.format(i))
            acti_op = ActiLayer(func=self.acti_func,
                                regularizer=self.regularizers['w'],
                                name='acti_{}'.format(i))
            conv_op = ConvLayer(n_output_chns=self.n_output_chns,
                                kernel_size=self.kernels[i],
                                stride=self.strides[i],
                                dilation=self.dilation_rates[i],
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                name='conv_{}'.format(i))
            output_tensor = conv_op(output_tensor)
            output_tensor = acti_op(output_tensor)
            output_tensor = bn_op(output_tensor, is_training)  # Construct operation first and then connect them.
        # make residual connections
        if self.with_res:
            # The input is directly added to the output.
            output_tensor = ElementwiseLayer('SUM')(output_tensor, input_tensor)
        return output_tensor


class TensorSliceLayer(TrainableLayer):
    def __init__(self, margin = 1, regularizer=None, name='tensor_extract'):
        self.layer_name = name
        super(TensorSliceLayer, self).__init__(name=self.layer_name)
        self.margin = margin

    def layer_op(self, input_tensor):
        input_shape = input_tensor.get_shape().as_list()
        begin = [0]*len(input_shape)
        begin[1] = self.margin
        size = input_shape
        end = [y - x for x, y in zip(begin[1:], size[1:])]
        begin = begin[1:]
        output_tensor = input_tensor[:, begin[0]:end[0], begin[1]:end[1], begin[2]:end[2], begin[3]:end[3]]
        return output_tensor


class ImageResize(TrainableLayer):
    def __init__(self, name='image_resize'):
        self.layer_name = name
        super(ImageResize, self).__init__(name=self.layer_name)

    def layer_op(self, image_tensor, feature_tensor):
        feature_shape = feature_tensor.get_shape().as_list()
        image_tensor = tf.transpose(image_tensor[..., 0], perm=[0, 2, 3, 1])
        image_tensor = tf.image.resize_images(image_tensor, feature_shape[-3:-1])
        image_tensor = tf.transpose(image_tensor, perm=[0, 3, 1, 2])
        image_tensor = tf.expand_dims(image_tensor, axis=-1)
        return image_tensor

if __name__ == '__main__':
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    net = DMapNet(num_classes=2)
    predicty = net(x, is_training = True)
    print(x)
    print(predicty)
