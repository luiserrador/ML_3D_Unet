from utils.layers import _check_inputs
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


def n_conv_block(x, n_convs=2, n_filters=8, ksize=3, padding='same', norm=None, activation='relu', depth=None,
                 name=None, **kwargs):
    """ n x Tensorflow Convolutions

    Parameters
    ----------
    x : tensor, np.array
        Input of the convolution block
    n_convs : scalar
        Number of convolutions, default=2
    n_filters : scalar
        Number os filters, default=8
    ksize : scalar or tuple of scalars
        Kernel size, default=3
    padding : str
        Padding parameter as on Tensorflow ('same', 'valid', ...), default='same'
    norm : str or None
        Normalization method - choose 'batch_norm', default=None
    activation : str or None
        Activation parameter as on Tensorflow ('relu', 'linear', ...), default='relu'
    depth : scalar
        Depth level on U-net, default=None
    name : str
        'encoder' or 'decoder' to identify compression and extension paths, default=None

    Returns
    -------
    x : tensor
        Output of the convolution block
    """

    for layer in range(n_convs):
        x = Conv3D(n_filters, ksize, padding=padding, name='{}_conv{}-{}'.format(name, depth, layer), **kwargs)(x)

        if norm == 'batch_norm':
            x = BatchNormalization(-1, name='{}_norm{}-{}'.format(name, depth, layer))(x)

        x = Activation(activation, name='{}_act{}-{}'.format(name, depth, layer))(x)

    return x


def downsample_block(x, n_convs=2, n_filters=8, ksize=3, padding='same', activation='relu', pooling='max',
                     norm=None, dropout=0, depth=None, **kwargs):
    """ Create Downsample Block for 3D-Unet:
    Apply n x Convolutions followed by Pooling and Dropout

    Parameters
    -----------
    x : tensor, np.array
        Input of the downsample block
    n_convs : scalar
        Number of convolutions
    n_filters : scalar
        Number os filters
    ksize : scalar or tuple of scalars
        Kernel size
    padding : str
        Padding parameter as on Tensorflow ('same', 'valid', ...)
    norm : str or None
        Normalization method - choose 'batch_norm'
    pooling : str
        Pooling method - 'avg' for AveragePooling, 'max' for MaxPooling
    dropout : float
        Dropout rate to be applied after pooling (float between 0.0 and 1.0)
    activation : str or None
        Activation parameter as on Tensorflow ('relu', 'linear', ...)
    depth : scalar
        Depth level on U-net

    Returns
    ----------
    conv_out : tensor
        output layer to concatenate on decoder
    x : tensor
        output of the downsample block after applying n_convs consecutive convolutions, pooling and dropout"""

    conv_out = n_conv_block(x, n_convs, n_filters, ksize, padding=padding, activation=activation, depth=depth,
                            norm=norm, **kwargs)

    if pooling == 'max':
        x = MaxPooling3D(pool_size=[2, 2, 2], padding=padding, name='encoder_maxpool{}'.format(depth))(conv_out)
    elif pooling == 'avg':
        x = AveragePooling3D(pool_size=[2, 2, 2], padding=padding, name='encoder_avgpool{}'.format(depth))(conv_out)

    if dropout > 0:
        x = Dropout(dropout, name='encoder_drop{}'.format(depth))(x)

    return conv_out, x


def upsample_block(layers_to_concat, x, n_convs=2, n_filters=8, ksize=3, padding='same', activation='relu',
                   norm=None, dropout=0, depth=None, upsampling=True, **kwargs):
    """ Upsample Block for 3D-Unet:
    Apply UpSampling/Conv3DTranspose followed by concatenation of contraction path connections, dropout and ConvBlock

    Parameters
    -----------
    layers_to_concat : dict
        Dictionary with the layer to concatenate with the upsampled x
    x : tensor
        Input of the upsample block
    n_convs : scalar
        Number of convolutions, default=2
    n_filters : scalar
        Number os filters, deafult=8
    ksize : scalar or tuple of scalars
        Kernel size, default=3
    padding : str
        Padding parameter as on Tensorflow ('same', 'valid', ...), default='same'
    activation : str or None
        Activation parameter as on Tensorflow ('relu', 'linear', ...), default='relu'
    norm : str or None
        Normalization method - choose 'batch_norm', default=None
    dropout : float
        Dropout rate to be applied after upsampling/deconvolution (float between 0.0 and 1.0), default=0.0
    depth : scalar
        Depth level on U-net, default=None
    upsampling : boolean
        Whether to use UpSampling (True) or Conv3DTranspose (False), default=True

    Returns
    -------
    x : tensor
        Output of the upsample block
    """
    if upsampling:
        x = UpSampling3D(size=[2, 2, 2], name='decoder_up{}'.format(depth))(x)
        x = Concatenate(name='decoder_concat{}'.format(depth))([x, layers_to_concat[str(depth)]])

        if dropout > 0 and depth != 0:
            x = Dropout(dropout, name='decoder_drop{}'.format(depth))(x)

        x = n_conv_block(x, n_convs, n_filters, ksize, padding=padding, activation=activation, depth=depth, norm=norm,
                         name='decoder', **kwargs)

    else:
        x = Conv3DTranspose(n_filters, kernel_size=2, strides=2, padding=padding, name='decoder_conv{}'.format(depth))(
            x)

        if norm == 'batch_norm':
            x = BatchNormalization(-1, name='decoder_norm{}'.format(depth))(x)

        x = Activation(activation, name='decoder_act{}'.format(depth))(x)
        x = Concatenate(name='decoder_concat{}'.format(depth))([x, layers_to_concat[str(depth)]])

        if dropout > 0 and depth != 0:
            x = Dropout(dropout, name='decoder_drop{}'.format(depth))(x)

        x = n_conv_block(x, n_convs, n_filters, ksize, padding=padding, activation=activation, depth=depth, norm=norm,
                         name='decoder', **kwargs)

    return x


def encoder_unet3d(x, n_convs=2, n_filters=8, ksize=3, padding='same', activation='relu', pooling='max',
                   norm=None, dropout=[0], depth=None, **kwargs):
    """ Encoder Block for 3D-Unet:
    Apply n x Convolutions followed by Downsample Blocks

    Parameters
    -----------
    x : tensor, np.array
        Input of the encoder block
    n_convs : scalar
        Number of convolutions, default=2
    n_filters : scalar
        Number os filters, default=8
    ksize : scalar or tuple of scalars
        Kernel size, default=3
    padding : str
        Padding parameter as on Tensorflow ('same', 'valid', ...), default='same'
    activation : str or None
        Activation parameter as on Tensorflow ('relu', 'linear', ...), default='relu'
    pooling : str
        Pooling method - 'avg' for AveragePooling, 'max' for MaxPooling, default='avg'
    norm : str or None
        Normalization method - choose 'batch_norm', default=None
    dropout : tuple, np.array
        Dropout rate to be applied after pooling (between 0.0 and 1.0)
    depth : scalar
        Depth level on U-net

    Returns
    ----------
    x : tensor
        output of the encoder block"""

    n_filters, ksize, dropout, depth = _check_inputs(n_filters, depth, dropout, ksize)

    x = n_conv_block(x, n_convs, n_filters[0], ksize[0], padding=padding, activation=activation, depth=0, norm=norm,
                     name='encoder', **kwargs)

    layers_to_concat = dict()

    for level in range(depth - 1):
        layers_to_concat['{}'.format(level)], x = downsample_block(x, n_convs, n_filters[level], ksize[level], padding,
                                                                   activation, pooling, norm, dropout[level],
                                                                   depth=level + 1, name='encoder', **kwargs)

    x = n_conv_block(x, n_convs, n_filters[-1], ksize[-1], padding=padding, activation=activation, depth=depth + 1,
                     name='bottleneck', norm=norm, **kwargs)

    return layers_to_concat, x


def decoder_unet3d(layers_to_concat, x, n_convs=2, n_filters=8, ksize=3, padding='same', activation='relu',
                   upsampling=True, norm=None, dropout=[0], depth=1, **kwargs):
    """ Decoder Block for 3D-Unet:
    Series of Upsample Blocks

    Parameters
    -----------
    layers_to_concat : dict
        Dictionary with the layers to concatenate with the upsampled x
    x : tensor
        Input of the upsample block
    n_convs : scalar
        Number of convolutions, default=2
    n_filters : scalar
        Number os filters, deafult=8
    ksize : scalar or tuple of scalars
        Kernel size, default=3
    padding : str
        Padding parameter as on Tensorflow ('same', 'valid', ...), default='same'
    activation : str or None
        Activation parameter as on Tensorflow ('relu', 'linear', ...), default='relu'
    upsampling : boolean
        Whether to use UpSampling (True) or Conv3DTranspose (False), default=True
    norm : str or None
        Normalization method - choose 'batch_norm', default=None
    dropout : float
        Dropout rate to be applied after upsampling/deconvolution (float between 0.0 and 1.0), default=0.0
    depth : scalar
        Depth level on U-net, default=None

    Returns
    -------
    x : tensor
        Output of the decoder block
    """

    n_filters, ksize, dropout, depth = _check_inputs(n_filters, depth, dropout, ksize)

    for level in range(depth - 2, -1, -1):
        x = upsample_block(layers_to_concat, x, n_convs, n_filters[level], ksize[level], padding, activation,
                           norm, dropout[level], level, upsampling, **kwargs)

    return x


def create_unet3d(input_shape, n_convs=2, n_filters=None, ksize=None, padding='same', activation='relu', pooling='max',
                  norm=None, dropout=None, depth=None, upsampling=True, **kwargs):
    """ Creates a Tensorflow Model of a 3D-Unet

    Parameters
    -----------
    input_shape : tensor, tuple
        Shape of 3D-Unet input
    n_convs : scalar
        Number of convolutions, default=2
    n_filters : scalar
        Number os filters
    ksize : scalar or tuple of scalars
        Kernel size
    padding : str
        Padding parameter as on Tensorflow ('same', 'valid', ...)
    activation : str or None
        Activation parameter as on Tensorflow ('relu', 'linear', ...)
    pooling : str
        Pooling method - 'avg' for AveragePooling, 'max' for MaxPooling
    norm : str or None
        Normalization method - choose 'batch_norm'
    dropout : float
        Dropout rate to be applied after upsampling/deconvolution (float between 0.0 and 1.0)
    depth : scalar
        Depth of U-net
    upsampling : boolean
        Whether to use UpSampling (True) or Conv3DTranspose (False), default=True

    Returns
    -------
    out : tensorflow.keras.models.Model
        3D-Unet Model"""

    n_filters, ksize, dropout, depth = _check_inputs(n_filters, depth, dropout, ksize)

    inputs = Input(shape=input_shape)

    layers_to_concat, x = encoder_unet3d(inputs, n_convs, n_filters, ksize, padding, activation, pooling,
                                         norm, dropout, depth, **kwargs)

    x = decoder_unet3d(layers_to_concat, x, n_convs, n_filters, ksize, padding, activation, upsampling,
                       norm, dropout, depth, **kwargs)

    x = Conv3D(1, 1, padding=padding, name='last_logits')(x)
    output = Activation('sigmoid', name='output')(x)

    return Model(inputs=inputs, outputs=output, name='3D-Unet'.format(input_shape))