from keras import backend as K
import keras
from keras.models import Model
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize
from keras import layers
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Lambda
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs

WEIGHTS_PATH_NO_TOP = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(include_top=True, weights='imagenet',
             input_tensor=None, input_shape=None,
             pooling=None,
             classes=1000):
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=197,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet50')

    # load weights
    if weights == 'imagenet':
        model.load_weights(WEIGHTS_PATH_NO_TOP)
        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='avg_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1000')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model


def get_random_image(img_groups, group_names, gid):
    gname = group_names[gid]
    photos = img_groups[gname]
    pid = np.random.choice(np.arange(len(photos)), size=1)[0]
    pname = photos[pid]
    return gname + pname + ".jpg"

def create_triples(image_dir):
    img_groups = {}
    for img_file in os.listdir(image_dir):
        prefix, suffix = img_file.split(".")
        gid, pid = prefix[0:4], prefix[4:]

        if gid in img_groups.keys():
            img_groups[gid].append(pid)
        else:
            img_groups[gid] = [pid]
    pos_triples, neg_triples = [], []

    for key in img_groups.keys():
        triples = [(key + x[0] + ".jpg", key + x[1] + ".jpg", str(int(key)+1 if int(key)<1495 else int(key)-3)+'01'+'.jpg')
                 for x in itertools.combinations(img_groups[key], 2)]
        pos_triples.extend(triples)

    return pos_triples

def triplet_loss(y_true, y_pred):
        ''' hard margin
        margin = K.constant(1)
        '''
        ''' soft margin'''
        dist = y_pred[:,0,0] - y_pred[:,1,0]
        margin = keras.backend.log(1+keras.backend.exp(dist))
        return K.mean(K.maximum(K.constant(0), K.square(y_pred[:,0,0]) - K.square(y_pred[:,1,0]) + margin))

def accuracy(y_true, y_pred):
    return K.mean(y_pred[:,0,0] < y_pred[:,1,0])

def l2Norm(x):
    return  K.l2_normalize(x, axis=-1)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

IMAGE_DIR = 'eval'
triples_data = create_triples(IMAGE_DIR)


dim = 5
h = 224
w= 224
anchor =np.zeros((dim,h,w,3))
positive =np.zeros((dim,h,w,3))
negative =np.zeros((dim,h,w,3))

for n,val in enumerate(triples_data[:5]):
    image_anchor = plt.imread(os.path.join(IMAGE_DIR, val[0]))
    image_anchor = imresize(image_anchor, (h, w))    
    image_anchor = image_anchor.astype("float32")
    #image_anchor = image_anchor/255.
    image_anchor = keras.applications.resnet50.preprocess_input(image_anchor, data_format='channels_last')
    anchor[n] = image_anchor

    image_positive = plt.imread(os.path.join(IMAGE_DIR, val[1]))
    image_positive = imresize(image_positive, (h, w))
    image_positive = image_positive.astype("float32")
    #image_positive = image_positive/255.
    image_positive = keras.applications.resnet50.preprocess_input(image_positive, data_format='channels_last')
    positive[n] = image_positive

    image_negative = plt.imread(os.path.join(IMAGE_DIR, val[2]))
    image_negative = imresize(image_negative, (h, w))
    image_negative = image_negative.astype("float32")
    #image_negative = image_negative/255.
    image_negative = keras.applications.resnet50.preprocess_input(image_negative, data_format='channels_last')
    negative[n] = image_negative


#Why is it random int? Because it is useless...
Y_train = np.random.randint(2, size=(1,2,dim)).T


resnet_input = Input(shape=(h,w,3))
# download weights from the internet
resnet_model = keras.applications.resnet50.ResNet50(weights='imagenet', include_top = False, input_tensor=resnet_input)
# load the local weights
#resnet_model = ResNet50(weights='imagenet', include_top = False, input_tensor=resnet_input)


for layer in resnet_model.layers:
    layer.trainable = False  


net = resnet_model.output
net = Flatten(name='flatten')(net) 
net = Dense(128, activation='relu', name='embed')(net)
net = Dense(128, activation='relu', name='embed2')(net)
net = Dense(128, activation='relu', name='embed3')(net)
net = Dropout(0.5)(net)
net = Lambda(l2Norm, output_shape=[128])(net)

base_model = Model(resnet_model.input, net, name='resnet_model')

input_shape=(h,w,3)
input_anchor = Input(shape=input_shape, name='input_anchor')
input_positive = Input(shape=input_shape, name='input_pos')
input_negative = Input(shape=input_shape, name='input_neg')

net_anchor = base_model(input_anchor)
net_positive = base_model(input_positive)
net_negative = base_model(input_negative)

positive_dist = Lambda(euclidean_distance, name='pos_dist')([net_anchor, net_positive])
negative_dist = Lambda(euclidean_distance, name='neg_dist')([net_anchor, net_negative])

stacked_dists = Lambda( 
            lambda vects: K.stack(vects, axis=1),
            name='stacked_dists'
)([positive_dist, negative_dist])


model = Model([input_anchor, input_positive, input_negative], stacked_dists, name='triple_siamese')

# find better metrics
model.compile(optimizer="rmsprop", loss=triplet_loss,metrics=[accuracy])

model.fit([anchor, positive, negative], Y_train, epochs=2,  batch_size=2, validation_split=0.1)
model.save('triplet_loss_resnet50.h5')
output =  model.predict([anchor,positive,negative])
print(output)