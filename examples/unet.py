from utils.layers import create_unet3d_class
from utils.layers_func import create_unet3d


def _get_unet_depth3():
    """Create 3D-Unet with 3 depth levels with:
        - 2 convolution on each level (n_convs)
        - 8 features in convolution layers (n_filters)
        - kernel size of [3, 3, 3]
        - 'same' padding to maintain input size
        - max pooling
        - batch normalization after convolution
        - upsampling layer instead of transpose deconvolutions
        - 'relu' activations after normalization
        - no dropout"""

    unet_model_func = create_unet3d(input_shape=[32, 32, 32, 1],
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


if __name__ == '__main__':
    _get_unet_depth3()
