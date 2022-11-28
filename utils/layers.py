from tensorflow.keras.layers import *
import numpy as np
from tensorflow.keras.models import Model


class ConvBlock(Layer):
    """ n x Tensorflow Convolution Layer for 3D-Unet

    Parameters
    -----------
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
    activation : str or None
        Activation parameter as on Tensorflow ('relu', 'linear', ...)
    depth : scalar
        Depth level on U-net
    id : str
        'encoder' or 'decoder' to identify compression and extension paths
    """

    def __init__(self, n_convs, n_filters, ksize, padding, norm, activation, depth, id, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.n_convs = n_convs
        self.n_filters = n_filters
        self.ksize = ksize
        self.padding = padding
        self.norm = norm
        self.activation = activation
        self.depth = depth
        self.id = id
        self.conv_layers = dict()
        self.act_layers = dict()
        self.norm_layers = dict()

        for layer in range(self.n_convs):
            self.conv_layers[str(layer)] = Conv3D(self.n_filters, self.ksize, padding=self.padding,
                                                  name='{}_conv{}-{}'.format(self.id, self.depth, layer),
                                                  **kwargs)

            if self.norm == 'batch_norm':
                self.norm_layers[str(layer)] = BatchNormalization(-1, name='{}_norm{}-{}'.format(self.id, self.depth,
                                                                                                 layer))

            self.act_layers[str(layer)] = Activation(self.activation, name='{}_act{}-{}'.format(self.id, self.depth,
                                                                                                layer))

    def call(self, inputs, training=None, **kwargs):

        x = inputs

        for layer in range(self.n_convs):
            x = self.conv_layers[str(layer)](x)

            if self.norm == 'batch_norm':
                x = self.norm_layers[str(layer)](x)

            x = self.act_layers[str(layer)](x)

        return x

    def get_config(self):

        return dict(n_convs=self.n_convs,
                    n_filters=self.n_filters,
                    ksize=self.ksize,
                    padding=self.padding,
                    norm=self.norm,
                    activation=self.activation,
                    depth=self.depth,
                    id=self.id,
                    # **super(ConvBlock, self).get_config(),
                    )


class DownsampleBlock(Layer):
    """ Downsample Block for 3D-Unet:
    Apply n x Convolutions followed by Pooling and Dropout

    Parameters
    -----------
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
    id : str
        'encoder' or 'decoder' to identify compression and extension paths
    """

    def __init__(self, n_convs, n_filters, ksize, padding, norm, pooling, dropout, activation, depth, id,
                 **kwargs):
        super(DownsampleBlock, self).__init__(**kwargs)
        self.n_convs = n_convs
        self.n_filters = n_filters
        self.ksize = ksize
        self.padding = padding
        self.norm = norm
        self.pooling = pooling
        self.dropout = dropout
        self.activation = activation
        self.depth = depth
        self.id = id

        self.first_conv = ConvBlock(n_convs, n_filters, ksize, padding, norm, activation, depth, id)
        if pooling == 'max':
            self.pool = MaxPooling3D(pool_size=[2, 2, 2], padding=padding, name='encoder_maxpool{}'.format(depth))
        elif pooling == 'avg':
            self.pool = AveragePooling3D(pool_size=[2, 2, 2], padding=padding, name='encoder_avgpool{}'.format(depth))
        if self.dropout > 0:
            self.drop = Dropout(dropout, name='encoder_drop{}'.format(depth))

    def call(self, inputs, training=False, **kwargs):

        x = inputs

        layer_to_concat = self.first_conv(x)
        x = self.pool(layer_to_concat)
        if self.dropout > 0 and training:
            x = self.drop(x)

        return layer_to_concat, x

    def get_config(self):

        return dict(n_convs=self.n_convs,
                    n_filters=self.n_filters,
                    ksize=self.ksize,
                    padding=self.padding,
                    norm=self.norm,
                    pooling=self.pooling,
                    dropout=self.dropout,
                    activation=self.activation,
                    depth=self.depth,
                    id=self.id,
                    **super(DownsampleBlock, self).get_config(),
                    )


class EncoderBlock(Layer):
    """ Encoder Block for 3D-Unet:
    Apply n x Convolutions followed by Downsample Blocks

    Parameters
    -----------
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
    id : str
        'encoder' or 'decoder' to identify compression and extension paths
    """

    def __init__(self, n_convs, n_filters, ksize, padding, norm, pooling, dropout, activation, depth, id,
                 **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)

        n_filters, ksize, dropout, depth = _check_inputs(n_filters, depth, dropout, ksize)

        self.n_convs = n_convs
        self.n_filters = n_filters
        self.ksize = ksize
        self.padding = padding
        self.norm = norm
        self.pooling = pooling
        self.dropout = dropout
        self.activation = activation
        self.depth = depth
        self.id = id
        self.down = dict()

        self.first_conv = ConvBlock(n_convs, n_filters[0], ksize[0], padding, norm, activation, depth, id)

        for level in range(depth - 1):
            self.down[str(level)] = DownsampleBlock(n_convs, n_filters[level], ksize[level], padding, norm, pooling,
                                                    dropout[level], activation, level + 1, id, **kwargs)

        self.last_conv = ConvBlock(n_convs, n_filters[-1], ksize[-1], padding, norm, activation, depth='',
                                   id='bottleneck')

    def call(self, inputs, training=None, **kwargs):

        x = inputs
        x = self.first_conv(x)

        layers_to_concat = dict()

        for level in range(self.depth - 1):
            layers_to_concat['{}'.format(level)], x = self.down[str(level)](x)

        x = self.last_conv(x)

        return x, layers_to_concat

    def get_config(self):

        return dict(n_convs=self.n_convs,
                    n_filters=self.n_filters,
                    ksize=self.ksize,
                    padding=self.padding,
                    norm=self.norm,
                    pooling=self.pooling,
                    dropout=self.dropout,
                    activation=self.activation,
                    depth=self.depth,
                    id=self.id,
                    **super(EncoderBlock, self).get_config(),
                    )


class DeconvBlock(Layer):
    """ Deconvolution Block for 3D-Unet:
    Conv3DTranspose followed by Normalizaiton and Activation

    Parameters
    -----------
    n_filters : scalar
        Number os filters
    padding : str
        Padding parameter as on Tensorflow ('same', 'valid', ...)
    norm : str or None
        Normalization method - choose 'batch_norm'
    activation : str or None
        Activation parameter as on Tensorflow ('relu', 'linear', ...)
    depth : scalar
        Depth level on U-net
    id : str
        'encoder' or 'decoder' to identify compression and extension paths
    """

    def __init__(self, n_filters, padding, norm, activation, depth, id, **kwargs):
        super(DeconvBlock, self).__init__(**kwargs)

        self.n_filters = n_filters
        self.padding = padding
        self.norm = norm
        self.activation = activation
        self.depth = depth
        self.id = id

        self.deconv = Conv3DTranspose(n_filters, kernel_size=2, strides=2, padding=padding,
                                      name='{}_deconv_{}'.format(id, depth))

        if norm == 'batch_norm':
            self.norm_layer = BatchNormalization(-1, name='{}_norm_{}'.format(id, depth))

        self.act = Activation(activation, name='{}_act_{}'.format(id, depth))

    def call(self, inputs, training=None, **kwargs):

        x = inputs
        x = self.deconv(x)
        if self.norm == 'batch_norm':
            x = self.norm_layer(x)
        x = self.act(x)

        return x

    def get_config(self):

        return dict(n_filters=self.n_filters,
                    padding=self.padding,
                    norm=self.norm,
                    activation=self.activation,
                    depth=self.depth,
                    id=self.id,
                    **super(DeconvBlock, self).get_config(),
                    )


class UpsampleBlock(Layer):
    """ Upsample Block for 3D-Unet:
    Apply UpSampling/Conv3DTranspose followed by concatenation of contraction path connections, dropout and ConvBlock

    Parameters
    -----------
    n_convs : scalar
        Number of convolutions
    n_filters : scalar
        Number os filters
    ksize : scalar or tuple of scalars
        Kernel size
    padding : str
        Padding parameter as on Tensorflow ('same', 'valid', ...)
    activation : str or None
        Activation parameter as on Tensorflow ('relu', 'linear', ...)
    norm : str or None
        Normalization method - choose 'batch_norm'
    dropout : float
        Dropout rate to be applied after upsampling/deconvolution (float between 0.0 and 1.0)
    depth : scalar
        Depth level on U-net
    upsampling : boolean
        Whether to use UpSampling (True) or Conv3DTranspose (False)
    id : str
        'encoder' or 'decoder' to identify compression and extension paths
    """

    def __init__(self, n_convs, n_filters, ksize, padding, activation, norm, dropout, depth, upsampling, id, **kwargs):
        super(UpsampleBlock, self).__init__(**kwargs)

        self.n_convs = n_convs
        self.upsampling = upsampling
        self.n_filters = n_filters
        self.ksize = ksize
        self.padding = padding
        self.norm = norm
        self.dropout = dropout
        self.activation = activation
        self.depth = depth
        self.id = id

        if upsampling:
            self.up = UpSampling3D(size=[2, 2, 2], name='{}_up{}'.format(id, depth))
        else:
            self.up = DeconvBlock(n_filters, padding, norm, activation, depth, id)

        self.concat = Concatenate(name='decoder_concat{}'.format(depth))

        if dropout > 0:
            self.drop = Dropout(dropout, name='decoder_drop{}'.format(depth))

        self.last_conv = ConvBlock(n_convs, n_filters, ksize, padding, norm, activation, depth, id)

    def call(self, inputs, layer_to_concat, training=None, **kwargs):

        x = inputs
        x = self.up(x)
        x = self.concat([x, layer_to_concat])
        if self.dropout > 0 and training:
            x = self.drop(x)
        x = self.last_conv(x)

        return x

    def get_config(self):

        return dict(n_convs=self.n_convs,
                    upsampling=self.upsampling,
                    n_filters=self.n_filters,
                    ksize=self.ksize,
                    padding=self.padding,
                    norm=self.norm,
                    dropout=self.dropout,
                    activation=self.activation,
                    depth=self.depth,
                    id=self.id,
                    **super(UpsampleBlock, self).get_config(),
                    )


class DecoderBlock(Layer):
    """ Decoder Block for 3D-Unet:
    Upsample Blocks followed by a ConvBlock (last convolution)

    Parameters
    -----------
    n_convs : scalar
        Number of convolutions
    n_filters : scalar
        Number os filters
    ksize : scalar or tuple of scalars
        Kernel size
    padding : str
        Padding parameter as on Tensorflow ('same', 'valid', ...)
    activation : str or None
        Activation parameter as on Tensorflow ('relu', 'linear', ...)
    norm : str or None
        Normalization method - choose 'batch_norm'
    dropout : float
        Dropout rate to be applied after upsampling/deconvolution (float between 0.0 and 1.0)
    depth : scalar
        Depth level on U-net
    upsampling : boolean
        Whether to use UpSampling (True) or Conv3DTranspose (False)
    id : str
        'encoder' or 'decoder' to identify compression and extension paths
    """

    def __init__(self, n_convs, n_filters, ksize, padding, activation, norm, dropout, depth, upsampling, id, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)

        n_filters, ksize, dropout, depth = _check_inputs(n_filters, depth, dropout, ksize)

        self.n_convs = n_convs
        self.upsampling = upsampling
        self.n_filters = n_filters
        self.ksize = ksize
        self.padding = padding
        self.norm = norm
        self.dropout = dropout
        self.activation = activation
        self.depth = depth
        self.id = id
        self.up = dict()

        for level in range(depth - 2, -1, -1):
            self.up[str(level)] = UpsampleBlock(n_convs, n_filters[level], ksize[level], padding, activation, norm,
                                                dropout[level], depth=level, upsampling=upsampling, id=id)

        self.last_conv = Conv3D(1, 1, padding=padding, name='last_logits')

    def call(self, inputs, layer_to_concat, training=None, **kwargs):

        x = inputs
        for level in range(self.depth - 2, -1, -1):
            x = self.up[str(level)](x, layer_to_concat[str(level)])
        x = self.last_conv(x)

        return x

    def get_config(self):

        return dict(n_convs=self.n_convs,
                    upsampling=self.upsampling,
                    n_filters=self.n_filters,
                    ksize=self.ksize,
                    padding=self.padding,
                    norm=self.norm,
                    dropout=self.dropout,
                    activation=self.activation,
                    depth=self.depth,
                    id=self.id,
                    **super(DecoderBlock, self).get_config(),
                    )


class Unet3D_layer(Layer):
    """ Tensorflow Layer of 3D-Unet:
    Returns one Layer with a 3D-Unet structure

    Parameters
    -----------
    n_convs : scalar
        Number of convolutions
    n_filters : scalar
        Number os filters
    ksize : scalar or tuple of scalars
        Kernel size
    padding : str
        Padding parameter as on Tensorflow ('same', 'valid', ...)
    pooling : str
        Pooling method - 'avg' for AveragePooling, 'max' for MaxPooling
    norm : str or None
        Normalization method - choose 'batch_norm'
    dropout : float
        Dropout rate to be applied after upsampling/deconvolution (float between 0.0 and 1.0)
    upsampling : boolean
        Whether to use UpSampling (True) or Conv3DTranspose (False)
    activation : str or None
        Activation parameter as on Tensorflow ('relu', 'linear', ...)
    depth : scalar
        Depth of U-net
    """

    def __init__(self, n_convs, n_filters, ksize, padding, pooling, norm, dropout, upsampling, activation, depth,
                 **kwargs):
        super(Unet3D_layer, self).__init__(**kwargs)

        n_filters, ksize, dropout, depth = _check_inputs(n_filters, depth, dropout, ksize)

        self.n_convs = n_convs
        self.n_filters = n_filters
        self.ksize = ksize
        self.padding = padding
        self.pooling = pooling
        self.norm = norm
        self.dropout = dropout
        self.upsampling = upsampling
        self.activation = activation
        self.depth = depth

        self.encoder = EncoderBlock(n_convs, n_filters, ksize, padding, norm, pooling, dropout, activation, depth,
                                    id='encoder')
        self.decoder = DecoderBlock(n_convs, n_filters, ksize, padding, activation, norm, dropout, depth, upsampling,
                                    id='decoder')

    def call(self, inputs, training=None, **kwargs):
        x = inputs
        x, layer_to_concat = self.encoder(x)
        x = self.decoder(x, layer_to_concat)

        return x

    def get_config(self):
        return dict(n_convs=self.n_convs,
                    n_filters=self.n_filters,
                    ksize=self.ksize,
                    padding=self.padding,
                    pooling=self.pooling,
                    norm=self.norm,
                    dropout=self.dropout,
                    upsampling=self.upsampling,
                    activation=self.activation,
                    depth=self.depth,
                    **super(Unet3D_layer, self).get_config(),
                    )


class Unet3D(Model):
    """ Tensorflow Model of 3D-Unet:
    Returns one Model with a 3D-Unet Layer

    Parameters
    -----------
    n_convs : scalar
        Number of convolutions
    n_filters : scalar
        Number os filters
    ksize : scalar or tuple of scalars
        Kernel size
    padding : str
        Padding parameter as on Tensorflow ('same', 'valid', ...)
    pooling : str
        Pooling method - 'avg' for AveragePooling, 'max' for MaxPooling
    norm : str or None
        Normalization method - choose 'batch_norm'
    dropout : float
        Dropout rate to be applied after upsampling/deconvolution (float between 0.0 and 1.0)
    upsampling : boolean
        Whether to use UpSampling (True) or Conv3DTranspose (False)
    activation : str or None
        Activation parameter as on Tensorflow ('relu', 'linear', ...)
    depth : scalar
        Depth of U-net
    """

    def __init__(self, n_convs, n_filters, ksize, padding, pooling, norm, dropout, upsampling, activation, depth,
                 **kwargs):
        super(Unet3D, self).__init__(**kwargs)

        n_filters, ksize, dropout, depth = _check_inputs(n_filters, depth, dropout, ksize)

        self.n_convs = n_convs
        self.n_filters = n_filters
        self.ksize = ksize
        self.padding = padding
        self.pooling = pooling
        self.norm = norm
        self.dropout = dropout
        self.upsampling = upsampling
        self.activation = activation
        self.depth = depth

        self.unet = Unet3D_layer(n_convs, n_filters, ksize, padding, pooling, norm, dropout, upsampling, activation,
                                 depth)

    def call(self, inputs, training=None, **kwargs):
        x = self.unet(inputs)

        return x


def create_unet3d_class(input_shape, n_convs=2, n_filters=None, ksize=None, padding='same', pooling='avg', norm=None,
                        dropout=None, upsampling=True, activation='relu', depth=None):
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
    pooling : str
        Pooling method - 'avg' for AveragePooling, 'max' for MaxPooling
    norm : str or None
        Normalization method - choose 'batch_norm'
    dropout : float
        Dropout rate to be applied after upsampling/deconvolution (float between 0.0 and 1.0)
    upsampling : boolean
        Whether to use UpSampling (True) or Conv3DTranspose (False), default=True
    activation : str or None
        Activation parameter as on Tensorflow ('relu', 'linear', ...)
    depth : scalar
        Depth of U-net

    Returns
    -------
    out : tensorflow.keras.models.Model
        3D-Unet Model"""

    layer = Unet3D_layer(n_convs,
                         n_filters,
                         ksize,
                         padding,
                         pooling,
                         norm,
                         dropout,
                         upsampling,
                         activation,
                         depth)

    input_model = Input(shape=input_shape)
    output_model = layer(input_model)

    return Model(inputs=input_model, outputs=output_model)


def _check_inputs(n_filters, depth, dropout, ksize):
    assert depth or n_filters, "Define n_filters"
    if not n_filters:
        n_filters = np.ones(depth) * 8
    elif not depth:
        depth = len(n_filters)

    if np.isscalar(dropout):
        dropout = np.ones(depth - 1, dtype=int) * dropout
    else:
        try:
            if len(dropout) < depth - 1:
                dropout = np.append(dropout, np.zeros(depth - len(dropout) - 1))
        except:
            dropout = np.zeros(depth - 1)

    assert len(dropout) == depth - 1, "len(dropout) < depth"

    if np.isscalar(ksize):
        ksize = np.stack((np.ones(3, dtype=int) * ksize,) * depth, axis=0)
    else:
        try:
            if ksize.shape[1] == 3 and ksize.shape[0] == 1:
                ksize = np.stack((ksize,) * depth, axis=0)
        except:
            ksize = np.stack((np.ones(3, dtype=int) * 3,) * depth, axis=0)

    assert ksize.shape[0] == depth and ksize.shape[1] == 3, "ksize is a scalar or an array of shape: [depth, 3]"

    return n_filters, ksize, dropout, depth
