import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

class TensorSliceLayer(layers.Layer):
    def __init__(self, margin=1):
        super(TensorSliceLayer, self).__init__()
        self.margin = margin

    def call(self, inputs):
        input_shape = inputs.get_shape().as_list()
        begin = [0] * len(input_shape)
        begin[1] = self.margin
        size = input_shape
        end = [y - x for x, y in zip(begin[1:], size[1:])]
        begin = begin[1:]
        output_tensor = inputs[:, begin[0]:end[0], begin[1]:end[1], begin[2]:end[2], begin[3]:end[3]]

        return output_tensor

class ResBlock(layers.Layer):
    def __init__(self,
                 n_output_chns=None,
                 kernels=None,  # [[1, 3, 3], [1, 3, 3]],
                strides=None,  # [[1, 1, 1], [1, 1, 1]],
                dilation_rates=None,  # [[1, 1, 1], [1, 1, 1]],
                activation='prelu',
                kernel_regularizer=None,
                bias_regularizer=None):
        super(ResBlock, self).__init__()
        # aa = tf.keras.Sequential()
        self.n_output_chns = n_output_chns
        self.activation = activation
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernels = [[1, 3, 3], [1, 3, 3]] if kernels is None else kernels
        self.strides = [[1, 1, 1], [1, 1, 1]] if strides is None else strides
        self.dilation_rates = [[1, 1, 1], [1, 1, 1]] if dilation_rates is None else dilation_rates
        self.block = tf.keras.Sequential()
        for i in range(len(self.kernels)):
            # create parameterised layers
            self.block.add(layers.Conv3D(filters=self.n_output_chns,
                                         kernel_size=self.kernels[i],
                                         strides=self.strides[i],
                                         dilation_rate=self.dilation_rates[i],
                                         padding="same",
                                         kernel_regularizer=self.kernel_regularizer,
                                         bias_regularizer=self.bias_regularizer))
            self.block.add(layers.Activation(activation=self.activation))
            self.block.add(layers.BatchNormalization())

    def call(self, inputs):

        return self.block(inputs) + inputs

def ImageResize(img, feature):
    feature_shape = feature.get_shape().as_list()
    image_tensor = tf.transpose(img[..., 0], perm=[0, 2, 3, 1])
    image_tensor = tf.image.resize(image_tensor, feature_shape[-3:-1])
    image_tensor = tf.transpose(image_tensor, perm=[0, 3, 1, 2])
    image_tensor = tf.expand_dims(image_tensor, axis=-1)

    return image_tensor

def DMapNetCompiled(input_size=(24, 128, 128, 1),
                    num_classes=None,
                    kernel_regularizer=None,
                    bias_regularizer=None,
                    activation='relu'):

    base_chns = [16, 32, 64, 64]
    block1_1 = ResBlock(base_chns[0],
                        kernels=[[1, 3, 3], [1, 3, 3]],
                        activation=activation,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer)

    block1_2 = ResBlock(base_chns[0],
                        kernels=[[1, 3, 3], [1, 3, 3]],
                        activation=activation,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer)

    block2_1 = ResBlock(base_chns[1],
                        kernels=[[1, 3, 3], [1, 3, 3]],
                        activation=activation,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer)

    block2_2 = ResBlock(base_chns[1],
                        kernels=[[1, 3, 3], [1, 3, 3]],
                        activation=activation,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer)

    block3_1 = ResBlock(base_chns[2],
                        kernels=[[1, 3, 3], [1, 3, 3]],
                        dilation_rates=[[1, 1, 1], [1, 1, 1]],
                        activation=activation,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer)

    block3_2 = ResBlock(base_chns[2],
                        kernels=[[1, 3, 3], [1, 3, 3]],
                        dilation_rates=[[1, 2, 2], [1, 2, 2]],
                        activation=activation,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer)

    block4_1 = ResBlock(base_chns[3],
                        kernels=[[1, 3, 3], [1, 3, 3]],
                        dilation_rates=[[1, 1, 1], [1, 1, 1]],  # [[1, 3, 3], [1, 3, 3]],
                        activation=activation,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer)

    block4_2 = ResBlock(base_chns[3],
                        kernels=[[1, 3, 3], [1, 3, 3]],
                        dilation_rates=[[1, 2, 2], [1, 2, 2]],
                        activation=activation,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer)

    fuse1 = layers.Conv3D(base_chns[0],
                          kernel_size=[3, 1, 1],  # Convolution on intra layers
                          padding='valid',
                          kernel_regularizer=kernel_regularizer,
                          bias_regularizer=kernel_regularizer)

    downsample1 = layers.Conv3D(base_chns[0],
                                kernel_size=[1, 3, 3],
                                strides=[1, 2, 2],
                                padding='same',
                                kernel_regularizer=kernel_regularizer,
                                bias_regularizer=bias_regularizer,
                                activation=activation)

    fuse2 = layers.Conv3D(base_chns[1],
                          kernel_size=[3, 1, 1],
                          padding='valid',
                          kernel_regularizer=kernel_regularizer,
                          bias_regularizer=bias_regularizer,
                          activation=activation)

    downsample2 = layers.Conv3D(base_chns[1],
                                kernel_size=[1, 3, 3],
                                strides=[1, 2, 2],
                                padding='same',
                                kernel_regularizer=kernel_regularizer,
                                bias_regularizer=bias_regularizer,
                                activation=activation)

    fuse3 = layers.Conv3D(base_chns[2],
                          kernel_size=[3, 1, 1],
                          padding='valid',
                          kernel_regularizer=kernel_regularizer,
                          bias_regularizer=bias_regularizer,
                          activation=activation)

    fuse4 = layers.Conv3D(base_chns[3],
                          kernel_size=[3, 1, 1],
                          padding='valid',
                          kernel_regularizer=kernel_regularizer,
                          bias_regularizer=bias_regularizer,
                          activation=activation)

    feature_expand1 = layers.Conv3D(base_chns[1],  # Output channels
                                    kernel_size=[1, 1, 1],
                                    strides=[1, 1, 1],
                                    padding='SAME',
                                    kernel_regularizer=kernel_regularizer,
                                    bias_regularizer=bias_regularizer,
                                    activation=activation)

    feature_expand2 = layers.Conv3D(base_chns[2],
                                    kernel_size=[1, 1, 1],
                                    strides=[1, 1, 1],
                                    padding='SAME',
                                    kernel_regularizer=kernel_regularizer,
                                    bias_regularizer=bias_regularizer,
                                    activation=activation)

    feature_expand3 = layers.Conv3D(base_chns[3],
                                    kernel_size=[1, 1, 1],
                                    strides=[1, 1, 1],
                                    padding='SAME',
                                    kernel_regularizer=kernel_regularizer,
                                    bias_regularizer=bias_regularizer,
                                    activation=activation)

    centra_slice1 = TensorSliceLayer(margin=2)
    centra_slice2 = TensorSliceLayer(margin=1)

    pred_up1 = layers.Conv3DTranspose(num_classes,
                                      kernel_size=[1, 3, 3],
                                      strides=[1, 2, 2],
                                      padding='same',
                                      kernel_regularizer=kernel_regularizer,
                                      bias_regularizer=bias_regularizer,
                                      activation=activation)
    pred_up2_1 = layers.Conv3DTranspose(num_classes * 2,
                                        kernel_size=[1, 3, 3],
                                        strides=[1, 2, 2],
                                        padding='same',
                                        kernel_regularizer=kernel_regularizer,
                                        bias_regularizer=bias_regularizer,
                                        activation=activation)
    pred_up2_2 = layers.Conv3DTranspose(num_classes * 2,
                                        kernel_size=[1, 3, 3],
                                        strides=[1, 2, 2],
                                        padding='same',
                                        kernel_regularizer=kernel_regularizer,
                                        bias_regularizer=bias_regularizer,
                                        activation=activation)
    pred_up3_1 = layers.Conv3DTranspose(num_classes * 4,
                                        kernel_size=[1, 3, 3],
                                        strides=[1, 2, 2],
                                        padding='same',
                                        kernel_regularizer=kernel_regularizer,
                                        bias_regularizer=bias_regularizer,
                                        activation=activation)
    pred_up3_2 = layers.Conv3DTranspose(num_classes * 4,
                                        kernel_size=[1, 3, 3],
                                        strides=[1, 2, 2],
                                        padding='same',
                                        kernel_regularizer=kernel_regularizer,
                                        bias_regularizer=bias_regularizer,
                                        activation=activation)

    final_pred = layers.Conv3D(num_classes,  # Output two class: target and background
                               kernel_size=[1, 3, 3],
                               padding='same',  # Same: keep shape; valid: only get pixels with valid calculation.
                               kernel_regularizer=kernel_regularizer,
                               bias_regularizer=bias_regularizer)

    images = layers.Input(input_size)
    f1 = images
    f1 = block1_1(f1)
    f1 = block1_2(f1)
    f1 = fuse1(f1)
    f1 = downsample1(f1)

    img_resize1 = ImageResize(images, f1)
    img_resize1 = centra_slice2(img_resize1)
    f1 = tf.concat([img_resize1, f1], axis=4, name='concate')
    if (base_chns[0] != base_chns[1]):
        f1 = feature_expand1(f1)  # To keep same channel number in cascaded netblocks
    f1 = block2_1(f1)
    f1 = block2_2(f1)
    f1 = fuse2(f1)

    f2 = downsample2(f1)
    img_resize2 = ImageResize(images, f2)
    img_resize2 = centra_slice1(img_resize2)
    f2 = tf.concat([img_resize2, f2], axis=4, name='concate')
    if (base_chns[1] != base_chns[2]):
        f2 = feature_expand2(f2)
    f2 = block3_1(f2)
    f2 = block3_2(f2)
    f2 = fuse3(f2)

    f3 = f2
    if (base_chns[2] != base_chns[3]):
        f3 = feature_expand3(f3)
    f3 = block4_1(f3)
    f3 = block4_2(f3)
    f3 = fuse4(f3)

    p1 = centra_slice1(f1)
    p1 = pred_up1(p1)

    p2 = centra_slice2(f2)
    p2 = pred_up2_1(p2)
    p2 = pred_up2_2(p2)

    p3 = pred_up3_1(f3)
    p3 = pred_up3_2(p3)

    cat = tf.concat([p1, p2, p3], axis=4, name='concate')
    pred = final_pred(cat)
    # pred = tf.nn.softmax(pred)

    model = keras.Model(inputs=images, outputs=pred)

    return model


class ProgressBar(keras.callbacks.Callback):
    def __init__(self, progress, total_epoch):
        super(ProgressBar, self).__init__()
        self.process = progress
        self.total_epoch = total_epoch

    def on_epoch_end(self, epoch, logs=None):
        self.process.emit('Training', epoch, self.total_epoch)