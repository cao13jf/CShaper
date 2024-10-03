''''Library for 3D [Height, Width, Depth] volume transformations.
'''

# import dependency library
import random
import collections.abc as collections
import torch
import numpy as np
from scipy import ndimage
from skimage.transform import resize
from scipy.ndimage.morphology import distance_transform_edt

#  import user defined library
from Util.rand import Constant, Uniform, Gaussian

#=======================================
#    Basic class for transformation
#=======================================
class Base(object):
    #  sample random variables
    def sample(self, *shape):
        return shape

    #  define transformation actions
    def tf(self, img, k=0):  # By default, list as [raw, target]
        ## If reuse is set as True, no sampling is needed
        return img

    #  define how to call tf
    def __call__(self, imgs, dim=3, reuse_pros=False):  #
        # resampling parameters and set self properties
        if not reuse_pros:
            im = imgs if isinstance(imgs, np.ndarray) else imgs[0]
            shape = im.shape
            assert len(shape) == 3, "only support 3-dim data"
            self.sample(*shape)

        if isinstance(imgs, collections.Sequence):
            return [self.tf(x, k) for k, x in enumerate(imgs)]
        return self.tf(imgs)

    #  define print string
    def __str__(self):
        return 'Identity()' # used for eval()

Identity = Base


#=====================================
#      Rotation
#=====================================
class Rot90(Base):
    def __init__(self, axes=(0, 1)):
        self.axes = axes
        for a in self.axes:
            assert a > -1

    def sample(self, *shape):
        shape0 = list(shape)
        shape0[self.axes[0]] = shape[self.axes[1]]
        shape0[self.axes[1]] = shape[self.axes[0]]
        return shape0

    def tf(self, img, k=0):
        return np.rot90(img, axes=self.axes)

    def __str__(self):
        return "Rot90(axes=({}, {}))".format(*self.axes)


class RandomRotation(Base):
    def __init__(self, angle_spectrum=10):
        assert isinstance(angle_spectrum,int)
        self.angle_spectrum = angle_spectrum
        self.axes = [(1, 0), (2, 1), (2, 0)]

    def sample(self, *shape):
        self.axes_buffer = self.axes[np.random.choice(list(range(len(self.axes))))]
        self.angle_buffer = np.random.randint(-self.angle_spectrum, self.angle_spectrum)
        return list(shape)

    def tf(self, img, k=0):  #
        img = ndimage.rotate(img, self.angle_buffer, axes=self.axes_buffer, reshape=False, order=0, mode="constant", cval=0)
        return img

    def __str__(self):
        return "RandomRotion(axes={}, angle={}".format(self.axes_buffer, self.angle_buffer)


#===========================================
#   Flip
#===========================================
class Flip(Base):
    def __init__(self, axis=0):
        self.axis = axis

    def tf(self, img, k=0):
        return np.flip(img, self.axis)

    def __str__(self):
        return "Flip(axis={})".format(self.axis)

class RandomFlip(Base):
    def __init__(self, axis=0):
        self.axis = (0, 1, 2)
        self.x_buffer = None
        self.y_buffer = None
        self.z_buffer = None

    def sample(self, *shape):
        self.x_buffer = np.random.choice([True, False])
        self.y_buffer = np.random.choice([True, False])
        self.z_buffer = np.random.choice([True, False])
        return list(shape)

    def tf(self, img, k=0):
        if self.x_buffer:
            img = np.flip(img, axis=self.axis[0])
        if self.y_buffer:
            img = np.flip(img, axis=self.axis[1])
        if self.z_buffer:
            img = np.flip(img, axis=self.axis[2])

        return img


#============================================
#   crop when extend the volume
#============================================
class Centercrop(Base):
    def __init__(self, target_size):
        self.target_size = target_size
        self.buffer = None

    def sample(self, *shape):
        target_size = self.target_size
        start = [(rs - ts)//2 for rs, ts in zip(shape, target_size)]
        self.buffer = [slice(st, st + ts) for st, ts in zip(start, target_size)]
        return target_size

    def tf(self, img, k=0):
        return img[tuple(self.buffer)]

    def __str__(self):
        return "Centercrop({})".format(self.target_size)


class RandCrop(Centercrop):
    def sample(self, *shape):
        start = [random.randint(0, rs - ts) for rs, ts in zip(shape, self.target_size)]
        self.buffer = [slice(st, st + ts) for st, ts in zip(start, self.target_size)]
        return self.target_size

    def __str__(self):
        return "RandCrop({})".format(self.target_size)


#======================================================
#   Pad when smaller than the target
#======================================================
class Pad(Base):
    def __init__(self, target_size=[208, 288, 144]):
        self.target_size = target_size

    def sample(self, *shape):
        return self.target_size

    def tf(self, img, k=0):
        raw_shape = img.shape
        pad_slice = [i - j for i, j in zip(self.target_size, raw_shape)]
        self.px = tuple((i//2, i-i//2) for i in pad_slice)
        return np.pad(img, self.px, mode="constant")

    def __str__(self):
        return "Pad {}, {}, {}".format(*self.px)


#===========================================
#   change intensity
#===========================================
class RandomIntensityChange(Base):
    def __init__(self, factor):
        shift, scale = factor
        assert (shift > 0) and (scale > 0), "shift {} and scale {} must > 0".format(shift, scale)
        self.shift = shift
        self.scale = scale

    def tf(self, img, k=0):
        if k==0:
            shift_buffer = np.random.uniform(-self.shift, self.shift, size = list(img.shape))
            scale_factor = np.random.uniform(1.0 - self.scale, 1.0 + self.scale, size=list(img.shape))
            return img * scale_factor + shift_buffer
        else:
            return img

    def __str__(self):
        return "random intensity shift on the input MembAndNuc"


#   add noise
class Noise(Base):
    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def tf(self, img, k=0):
        if k == 1:
            return img
        shape = img.shape
        return img * np.exp(self.sigma * torch.randn(shape, dtype=torch.float32).numpy())

    def __str__(self):
        return "Noise()"


# Gaussian noise
class GaussianBlur(Base):
    def __init__(self, sigma=Constant(1.5)):
        self.sigma = sigma
        self.eps = 0.001

    def tf(self, img, k=0):
        if k == 1:
            return img
        sig = self.sigma.sample()
        if sig > self.eps:
            img = ndimage.gaussian_filter(img, sig)
        return img

    def __str__(self):
        return "GaussianBlur()"


#   binary mask to distance
class ContourEDT(Base):
    #  only applicable to binary MembAndNuc
    def __init__(self, d_threshold=15):
        self.d_threshold = d_threshold

    def tf(self, img, k=0):
        if k==1 and len(np.unique(img)) == 2:
            background_edt = distance_transform_edt(img == 0)
            background_edt[background_edt > self.d_threshold] = self.d_threshold
            reversed_edt = (self.d_threshold - background_edt) / self.d_threshold

            return reversed_edt.astype(np.float32)
        else:
            return img


#   Normalization
class Normalize(Base):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def tf(self, img, k=0):
        if k==1:
            return img
        img -= self.mean
        img = self.std
        return img

    def __str__(self):
        return "Normalize()"


class Resize(Base):
    def __init__(self, target_size=(205, 288, 144)):
        assert len(target_size) == 3, "Only support in-slice resize"  #
        self.target_size = target_size

    def tf(self, img, k=0):
        if k == 0:
            resized_stack = resize(img, self.target_size, mode='constant', cval=0, order=1,anti_aliasing=True)
        else:
            resized_stack = resize(img, self.target_size, mode='constant', cval=0, order=0, anti_aliasing=True)
        return resized_stack


#=============================================
#   randomly select operations from a series of operations
#=============================================
class RandSelect(Base):
    def __init__(self, prob=0.5, tf=None):
        self.prob = prob
        self.ops = tf if isinstance(tf, collections.Sequence) else (tf, )
        self.buff = None

    def sample(self, *shape):
        self.buff = random < self.prob

        if self.buff:
            for op in self.ops:
                shape = op.sample(*shape)
        return shape

    def tf(self, img, k=0):
        if self.buff:
            for op in self.ops:
                img = op.tf(img, k)
        return img

    def __str__(self):
        if len(self.ops) == 1:
            ops = str(self.ops[0])
        else:
            ops = "[{}]".format(','.join([str(op) for op in self.ops]))
        return "RandSelect({}, {})".format(self.prob, ops)


#=================================================
#   Compose different operations.
#=================================================
class Compose(Base):
    def __init__(self, ops):
        if not isinstance(ops, collections.Sequence):
            ops = ops,
        self.ops = ops

    def sample(self, *shape):
        for op in self.ops:
            shape = op.sample(*shape)

    def tf(self, img, k=0):
        #is_tensor = isinstance(img, torch.Tensor)
        #if is_tensor:
        #    img = img.numpy()

        for op in self.ops:
            # print(op,img.shape,k)
            img = op.tf(img, k) # do not use op(img) here

        #if is_tensor:
        #    img = np.ascontiguousarray(img)
        #    img = torch.from_numpy(img)
        return img

    def __str__(self):
        ops = ', '.join([str(op) for op in self.ops])
        return 'Compose([{}])'.format(ops)

#=================================================
#   Data format transformation
#=================================================
class ToNumpy(Base):
    def __init__(self):
        pass

    def tf(self, img, k=0):
        return img.numpy()

    def __str__(self):
        return "ToNumpy()"


class ToTensor(Base):
    def __init__(self):
        pass

    def tf(self, img, k=0):
        return torch.from_numpy(img)

    def __str__(self):
        return "ToTensor"


class TensorType(Base):
    def __init__(self, types):
        self.types = types

    def tf(self, img, k=0):
        return img.type(self.types[k])

    def __str__(self):
        s_types = ",".join([str(s) for s in self.types])
        return "TensorType(({}))".format(s_types)


class NumpyType(Base):
    def __init__(self, types):
        self.types = types

    def tf(self, img, k=0):
        return img.astype(self.types[k])  # k?

    def __str__(self):
        s_types = ",".join([str(s) for s in self.types])
        return "NumpyType(({}))".format(s_types)





