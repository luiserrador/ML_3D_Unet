import netron
import os
import unittest

import tensorflow as tf

from utils.layers import *
from utils.layers_func import *

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Comment to use GPU


def testConvBlock():
    """Test ConvBlock"""

    conv_block = ConvBlock(n_convs=2,
                           n_filters=8,
                           ksize=3,
                           padding='same',
                           norm='batch_norm',
                           activation='relu',
                           depth=4,
                           id=1)

    config = conv_block.get_config()
    new_conv_block = ConvBlock.from_config(config)

    assert new_conv_block.n_convs == conv_block.n_convs
    assert new_conv_block.n_filters == conv_block.n_filters
    assert new_conv_block.ksize == conv_block.ksize
    assert new_conv_block.padding == conv_block.padding
    assert new_conv_block.norm == conv_block.norm
    assert new_conv_block.activation == conv_block.activation
    assert new_conv_block.depth == conv_block.depth
    assert new_conv_block.id == conv_block.id

    input = Input(shape=[128, 128, 128, 2])
    x = conv_block(input)
    new_x = new_conv_block(input)
    x_func = n_conv_block(input,
                          n_convs=2,
                          n_filters=8,
                          ksize=3,
                          padding='same',
                          norm='batch_norm',
                          activation='relu',
                          depth=4,
                          name='encoder')

    model = Model(inputs=input, outputs=x)
    new_model = Model(inputs=input, outputs=new_x)
    model_func = Model(inputs=input, outputs=x_func)

    assert model.count_params() == new_model.count_params() == model_func.count_params()
    assert len(model.trainable_variables) == len(new_model.trainable_variables) == len(model_func.trainable_variables)
    assert len(model.non_trainable_variables) == len(new_model.non_trainable_variables) == len(
        model_func.non_trainable_variables)


def testDownsampleBlock():
    """Test DownsampleBlock"""

    down_block = DownsampleBlock(n_convs=2,
                                 n_filters=8,
                                 ksize=3,
                                 padding='same',
                                 norm='batch_norm',
                                 pooling='avg',
                                 dropout=0.5,
                                 activation='relu',
                                 depth=3,
                                 id='encoder')

    config = down_block.get_config()
    new_down_block = DownsampleBlock.from_config(config)

    assert new_down_block.n_convs == down_block.n_convs
    assert new_down_block.n_filters == down_block.n_filters
    assert new_down_block.ksize == down_block.ksize
    assert new_down_block.padding == down_block.padding
    assert new_down_block.norm == down_block.norm
    assert new_down_block.pooling == down_block.pooling
    assert new_down_block.dropout == down_block.dropout
    assert new_down_block.activation == down_block.activation
    assert new_down_block.depth == down_block.depth
    assert new_down_block.id == down_block.id

    input = Input(shape=[128, 128, 128, 2])
    x = down_block(input)
    new_x = new_down_block(input)
    x_func = downsample_block(input,
                              n_convs=2,
                              n_filters=8,
                              ksize=3,
                              padding='same',
                              norm='batch_norm',
                              pooling='avg',
                              dropout=0.5,
                              activation='relu',
                              depth=3,
                              name='encoder')

    model = Model(inputs=input, outputs=x)
    new_model = Model(inputs=input, outputs=new_x)
    model_func = Model(inputs=input, outputs=x_func)

    assert model.count_params() == new_model.count_params() == model_func.count_params()
    assert len(model.trainable_variables) == len(new_model.trainable_variables) == len(model_func.trainable_variables)
    assert len(model.non_trainable_variables) == len(new_model.non_trainable_variables) == len(
        model_func.non_trainable_variables)


def testEncoderBlock():
    """Test EncoderBlock"""

    encoder_block = EncoderBlock(n_convs=2,
                                 n_filters=[8, 16, 32],
                                 ksize=[[3, 3, 3], [3, 3, 3], [3, 3, 3]],
                                 padding='same',
                                 norm='batch_norm',
                                 pooling='avg',
                                 dropout=[0.5, 0.5],
                                 activation='relu',
                                 depth=3,
                                 id='encoder')

    config = encoder_block.get_config()
    new_encoder_block = EncoderBlock.from_config(config)

    assert new_encoder_block.n_convs == encoder_block.n_convs
    assert new_encoder_block.n_filters == encoder_block.n_filters
    assert np.array_equal(new_encoder_block.ksize, encoder_block.ksize)
    assert new_encoder_block.padding == encoder_block.padding
    assert new_encoder_block.norm == encoder_block.norm
    assert new_encoder_block.pooling == encoder_block.pooling
    assert new_encoder_block.dropout == encoder_block.dropout
    assert new_encoder_block.activation == encoder_block.activation
    assert new_encoder_block.depth == encoder_block.depth
    assert new_encoder_block.id == encoder_block.id

    input = Input(shape=[128, 128, 128, 2])
    x = encoder_block(input)
    new_x = new_encoder_block(input)
    x_func = encoder_unet3d(input,
                            n_convs=2,
                            n_filters=[8, 16, 32],
                            ksize=[[3, 3, 3], [3, 3, 3], [3, 3, 3]],
                            padding='same',
                            norm='batch_norm',
                            pooling='avg',
                            dropout=[0.5, 0.5],
                            activation='relu',
                            depth=3)

    model = Model(inputs=input, outputs=x)
    new_model = Model(inputs=input, outputs=new_x)
    model_func = Model(inputs=input, outputs=x_func)

    assert model.count_params() == new_model.count_params() == model_func.count_params()
    assert len(model.trainable_variables) == len(new_model.trainable_variables) == len(model_func.trainable_variables)
    assert len(model.non_trainable_variables) == len(new_model.non_trainable_variables) == len(
        model_func.non_trainable_variables)


def testDeconvBlock():
    """Test DeconvBlock"""

    deconv_block = DeconvBlock(n_filters=8,
                               padding='same',
                               norm='batch_norm',
                               activation='relu',
                               depth=1,
                               id='decoder')

    config = deconv_block.get_config()
    new_deconv_block = DeconvBlock.from_config(config)

    assert new_deconv_block.n_filters == deconv_block.n_filters
    assert new_deconv_block.padding == deconv_block.padding
    assert new_deconv_block.norm == deconv_block.norm
    assert new_deconv_block.activation == deconv_block.activation
    assert new_deconv_block.depth == deconv_block.depth
    assert new_deconv_block.id == deconv_block.id

    input = Input(shape=[128, 128, 128, 2])
    x = deconv_block(input)
    new_x = new_deconv_block(input)

    model = Model(inputs=input, outputs=x)
    new_model = Model(inputs=input, outputs=new_x)

    assert model.count_params() == new_model.count_params()
    assert len(model.trainable_variables) == len(new_model.trainable_variables)
    assert len(model.non_trainable_variables) == len(new_model.non_trainable_variables)


def testUpsampleBlock():
    """Test UpsampleBlock"""

    up_block = UpsampleBlock(n_convs=2,
                             n_filters=8,
                             ksize=[3, 3, 3],
                             padding='same',
                             activation='relu',
                             norm='batch_norm',
                             dropout=0.5,
                             depth=3,
                             upsampling=True,
                             id='encoder')

    config = up_block.get_config()
    new_up_block = UpsampleBlock.from_config(config)

    assert new_up_block.n_convs == up_block.n_convs
    assert new_up_block.upsampling == up_block.upsampling
    assert new_up_block.n_filters == up_block.n_filters
    assert new_up_block.ksize == up_block.ksize
    assert new_up_block.padding == up_block.padding
    assert new_up_block.norm == up_block.norm
    assert new_up_block.dropout == up_block.dropout
    assert new_up_block.activation == up_block.activation
    assert new_up_block.depth == up_block.depth
    assert new_up_block.id == up_block.id

    input_encoder = Input(shape=[128, 128, 128, 2])
    encoder = EncoderBlock(n_convs=2,
                           n_filters=[8, 16, 32],
                           ksize=[[3, 3, 3], [3, 3, 3], [3, 3, 3]],
                           padding='same',
                           norm='batch_norm',
                           pooling='avg',
                           dropout=[0.5, 0.5],
                           activation='relu',
                           depth=3,
                           id='encoder')

    input, layers_to_concat = encoder(input_encoder)
    x = up_block(input, layers_to_concat['1'])
    new_x = new_up_block(input, layers_to_concat['1'])

    model = Model(inputs=input_encoder, outputs=x)
    new_model = Model(inputs=input_encoder, outputs=new_x)

    assert model.count_params() == new_model.count_params()
    assert len(model.trainable_variables) == len(new_model.trainable_variables)
    assert len(model.non_trainable_variables) == len(new_model.non_trainable_variables)


def testDecoderBlock():
    """Test DecoderBlock"""

    decoder_block = DecoderBlock(n_convs=2,
                                 n_filters=[8, 16, 32, 64],
                                 ksize=3,
                                 padding='same',
                                 activation='relu',
                                 norm='batch_norm',
                                 dropout=[0.25, 0.5, 0.5],
                                 depth=4,
                                 upsampling=True,
                                 id='decoder')

    config = decoder_block.get_config()
    new_decoder_block = DecoderBlock.from_config(config)

    assert new_decoder_block.n_convs == decoder_block.n_convs
    assert new_decoder_block.upsampling == decoder_block.upsampling
    assert new_decoder_block.n_filters == decoder_block.n_filters
    assert np.array_equal(new_decoder_block.ksize, decoder_block.ksize)
    assert new_decoder_block.padding == decoder_block.padding
    assert new_decoder_block.norm == decoder_block.norm
    assert new_decoder_block.dropout == decoder_block.dropout
    assert new_decoder_block.activation == decoder_block.activation
    assert new_decoder_block.depth == decoder_block.depth
    assert new_decoder_block.id == decoder_block.id

    input_encoder = Input(shape=[128, 128, 128, 2])
    encoder = EncoderBlock(n_convs=2,
                           n_filters=[8, 16, 32, 64],
                           ksize=3,
                           padding='same',
                           norm='batch_norm',
                           pooling='avg',
                           dropout=[0.25, 0.5, 0.5],
                           activation='relu',
                           depth=4,
                           id='encoder')

    input, layers_to_concat = encoder(input_encoder)
    x = decoder_block(input, layers_to_concat)
    new_x = new_decoder_block(input, layers_to_concat)

    model = Model(inputs=input_encoder, outputs=x)
    new_model = Model(inputs=input_encoder, outputs=new_x)

    assert model.count_params() == new_model.count_params()
    assert len(model.trainable_variables) == len(new_model.trainable_variables)
    assert len(model.non_trainable_variables) == len(new_model.non_trainable_variables)


def testDecoderBlock():
    """Test DecoderBlock"""

    decoder_block = DecoderBlock(n_convs=2,
                                 n_filters=[8, 16, 32, 64],
                                 ksize=3,
                                 padding='same',
                                 activation='relu',
                                 norm='batch_norm',
                                 dropout=[0.25, 0.5, 0.5],
                                 depth=4,
                                 upsampling=True,
                                 id='decoder')

    config = decoder_block.get_config()
    new_decoder_block = DecoderBlock.from_config(config)

    assert new_decoder_block.n_convs == decoder_block.n_convs
    assert new_decoder_block.upsampling == decoder_block.upsampling
    assert new_decoder_block.n_filters == decoder_block.n_filters
    assert np.array_equal(new_decoder_block.ksize, decoder_block.ksize)
    assert new_decoder_block.padding == decoder_block.padding
    assert new_decoder_block.norm == decoder_block.norm
    assert new_decoder_block.dropout == decoder_block.dropout
    assert new_decoder_block.activation == decoder_block.activation
    assert new_decoder_block.depth == decoder_block.depth
    assert new_decoder_block.id == decoder_block.id

    input_encoder = Input(shape=[128, 128, 128, 2])
    encoder = EncoderBlock(n_convs=2,
                           n_filters=[8, 16, 32, 64],
                           ksize=3,
                           padding='same',
                           norm='batch_norm',
                           pooling='avg',
                           dropout=[0.25, 0.5, 0.5],
                           activation='relu',
                           depth=4,
                           id='encoder')

    input, layers_to_concat = encoder(input_encoder)
    x = decoder_block(input, layers_to_concat)
    new_x = new_decoder_block(input, layers_to_concat)

    model = Model(inputs=input_encoder, outputs=x)
    new_model = Model(inputs=input_encoder, outputs=new_x)

    assert model.count_params() == new_model.count_params()
    assert len(model.trainable_variables) == len(new_model.trainable_variables)
    assert len(model.non_trainable_variables) == len(new_model.non_trainable_variables)


def testUnet3D_layer():
    """Test DecoderBlock"""

    unet_layer = Unet3D_layer(n_convs=2,
                              n_filters=[8, 16, 32, 64],
                              ksize=[3, 3, 3],
                              padding='same',
                              pooling='avg',
                              norm='batch_norm',
                              dropout=[0.25, 0.5, 0.5],
                              upsampling=True,
                              activation='relu',
                              depth=4)

    config = unet_layer.get_config()
    new_unet_layer = Unet3D_layer.from_config(config)

    assert new_unet_layer.n_convs == unet_layer.n_convs
    assert new_unet_layer.upsampling == unet_layer.upsampling
    assert new_unet_layer.n_filters == unet_layer.n_filters
    assert np.array_equal(new_unet_layer.ksize, unet_layer.ksize)
    assert new_unet_layer.padding == unet_layer.padding
    assert new_unet_layer.norm == unet_layer.norm
    assert new_unet_layer.dropout == unet_layer.dropout
    assert new_unet_layer.activation == unet_layer.activation
    assert new_unet_layer.depth == unet_layer.depth

    input = Input(shape=[128, 128, 128, 2])
    x = unet_layer(input)
    new_x = new_unet_layer(input)

    model = Model(inputs=input, outputs=x)
    new_model = Model(inputs=input, outputs=new_x)

    assert model.count_params() == new_model.count_params()
    assert len(model.trainable_variables) == len(new_model.trainable_variables)
    assert len(model.non_trainable_variables) == len(new_model.non_trainable_variables)


def testUnet3D():
    """Test Unet3D"""

    input = Input(shape=[128, 128, 128, 2])
    model_unet = Unet3D(n_convs=2,
                        n_filters=[8, 16, 32, 64],
                        ksize=[3, 3, 3],
                        padding='same',
                        pooling='avg',
                        norm='batch_norm',
                        dropout=[0.25, 0.5, 0.5],
                        upsampling=True,
                        activation='relu',
                        depth=4)

    _ = model_unet(input)  # call to .build() model
    _ = model_unet(tf.zeros([1, 128, 128, 128, 2]), trainable=True)

    unet_model = create_unet3d_class(input_shape=[128, 128, 128, 2],
                                     n_convs=2,
                                     n_filters=[8, 16, 32, 64],
                                     ksize=[3, 3, 3],
                                     padding='same',
                                     pooling='avg',
                                     norm='batch_norm',
                                     dropout=[0.25, 0.5, 0.5],
                                     upsampling=True,
                                     activation='relu',
                                     depth=4)

    unet_model_func = create_unet3d(input_shape=[128, 128, 128, 2],
                                    n_convs=2,
                                    n_filters=[8, 16, 32, 64],
                                    ksize=[3, 3, 3],
                                    padding='same',
                                    pooling='avg',
                                    norm='batch_norm',
                                    dropout=[0.25, 0.5, 0.5],
                                    upsampling=True,
                                    activation='relu',
                                    depth=4)

    assert unet_model.count_params() == model_unet.count_params() == unet_model_func.count_params()
    assert len(unet_model.trainable_variables) == len(model_unet.trainable_variables) == \
           len(unet_model_func.trainable_variables)
    assert len(unet_model.non_trainable_variables) == len(model_unet.non_trainable_variables) == \
           len(unet_model_func.non_trainable_variables)


if __name__ == '__main__':
    unittest.main()