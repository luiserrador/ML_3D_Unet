from utils.layers import create_unet3d_class
from utils.layers_func import create_unet3d


def _get_unet_1():
    """Create 3D-Unet with 3 depth levels with:
        - 2 convolution on each level (n_convs)
        - 8 features in convolution layers (n_filters)
        - kernel size of [3, 3, 3] (ksize)
        - 'same' padding to maintain input size (padding)
        - max pooling (pooling)
        - batch normalization after convolution (norm)
        - upsampling layer instead of transpose deconvolutions (upsampling)
        - 'relu' activations after normalization (activation)
        - no dropout (dropout)"""

    unet_model_func = create_unet3d(input_shape=[32, 32, 32, 1],
                                    n_convs=2,
                                    n_filters=[8, 8, 8],  # len(n_filters) == depth
                                    ksize=[3, 3, 3],  # or [[3, 3, 3], [3, 3, 3], [3, 3, 3]]
                                    padding='same',
                                    pooling='max',
                                    norm='batch_norm',
                                    dropout=[0, 0],  # len(dropout) == depth-1
                                    upsampling=True,
                                    activation='relu',
                                    depth=3)
    # OR #

    unet_model = create_unet3d_class(input_shape=[32, 32, 32, 1],
                                     n_convs=2,
                                     n_filters=[8, 8, 8],
                                     ksize=[3, 3, 3],
                                     padding='same',
                                     pooling='max',
                                     norm='batch_norm',
                                     dropout=[0, 0],
                                     upsampling=True,
                                     activation='relu',
                                     depth=3)

    unet_model_func.summary()
    unet_model.summary()

    return


def _get_unet_2():
    """Create 3D-Unet with 3 depth levels with:
        - 2 convolution on each level
        - different number of features on different levels
        - different kernel sizes on different levels
        - 'same' padding to maintain input size
        - average pooling
        - no batch normalization after convolution
        - transpose deconvolutions instead of upsampling
        - 'linear' activations after normalization
        - dropout"""

    unet_model_func = create_unet3d(input_shape=[32, 32, 32, 1],
                                    n_convs=2,
                                    n_filters=[8, 8, 8],
                                    ksize=[[7, 7, 7], [5, 5, 5], [3, 3, 3]],
                                    padding='same',
                                    pooling='avg',
                                    norm=None,
                                    dropout=[0.5, 0.5],
                                    upsampling=False,
                                    activation='linear',
                                    depth=3)
    # OR #

    unet_model = create_unet3d_class(input_shape=[32, 32, 32, 1],
                                     n_convs=2,
                                     n_filters=[8, 8, 8],
                                     ksize=[[7, 7, 7], [5, 5, 5], [3, 3, 3]],
                                     padding='same',
                                     pooling='avg',
                                     norm=None,
                                     dropout=[0.5, 0.5],
                                     upsampling=False,
                                     activation='linear',
                                     depth=3)

    unet_model_func.summary()
    unet_model.summary()

    return


if __name__ == '__main__':
    _get_unet_1()
    _get_unet_2()
