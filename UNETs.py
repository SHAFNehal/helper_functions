# -*- coding: utf-8 -*-
"""
@author: Syed Hasib Akhter Faruqui
@email: syed-hasib-akhter.faruqui@utsa.edu
@web: www.shafnehal.com
Note: set of UNET Functions
"""
import tensorflow as tf

def UNET(image_shape = (128, 128, 1), filter_sizes = [64, 128, 256, 512, 1024]):
    # Apply double convolution and ReLU to input image/feature
    def double_convolution(x, n_filters):
        # Conv2D --> ReLU : Initializtion "Glourot/he normal"
        image_feature = tf.keras.layers.Conv2D(n_filters, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(x)
        # Conv2D --> ReLU : Initializtion "Glourot/he normal"
        image_feature = tf.keras.layers.Conv2D(n_filters, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(image_feature)
        return image_feature

    def downsample_features(x):
        image_downsampled = tf.keras.layers.MaxPool2D(2)(x)
        image_downsampled = tf.keras.layers.Dropout(0.3)(image_downsampled)
        return image_downsampled

    def double_conv_downsample(x, n_filters):
        to_pass_forward = double_convolution(x, n_filters)
        to_pass_down = downsample_features(to_pass_forward)
        return to_pass_forward, to_pass_down

    def upsample_concatenate(x, conv_features, n_filters):
        # Upsample the incoming feature
        upsampled = tf.keras.layers.Conv2DTranspose(n_filters, 3, 2, padding = 'same')(x)
        # Concatenate the features
        concatenated_features = tf.keras.layers.concatenate([upsampled, conv_features])
        # Dropout
        concatenated_features = tf.keras.layers.Dropout(0.3)(concatenated_features)

        return concatenated_features

    def upsample_double_conv(x, conv_features, n_filters):
        # Concatenate them
        concatenated_features = upsample_concatenate(x, conv_features, n_filters)
        # 2 X Conv2D --> ReLU
        pass_forward = double_convolution(concatenated_features, n_filters)
        return pass_forward

    # Inputes
    inputs = tf.keras.layers.Input(shape = image_shape)

    forward = []
    # stack and build the part 1
    for index, value in enumerate(filter_sizes[:-1]):
        # For the initial Layer
        if index ==0:
            to_pass_down = inputs
        else:
            pass

        to_pass_forward, to_pass_down = double_conv_downsample(to_pass_down, n_filters = value)
        # Store Forward Features
        forward.append(to_pass_forward)

    # Bottle Neck
    bottleneck = double_convolution(to_pass_down, n_filters = filter_sizes[-1])
    # Change the order of the features stored
    forward.reverse()

    # Upsampling Setup
    for index, value in enumerate(reversed(filter_sizes[:-1])):
        if index == 0:
            to_pass_forward = bottleneck
        else:
            pass
        to_pass_forward = upsample_double_conv(to_pass_forward, conv_features = forward[index], n_filters = value)

    outputs = tf.keras.layers.Conv2D(1, 1, padding = 'same', activation='sigmoid')(to_pass_forward) #

    return tf.keras.Model(inputs, outputs, name = 'UNET')


    # Attention UNET
# Writing brute force way. Inspired by other imlementations on GITHUB
# https://www.researchgate.net/publication/348403300/figure/fig1/AS:978994182750215@1610421810031/The-architecture-of-our-proposed-Attention-U-Net_W640.jpg
def attention_unet(image_shape = (128, 128, 1)):
    # Apply double convolution and ReLU to input image/feature
    def double_convolution(x, n_filters):
        # Conv2D --> ReLU : Initializtion "Glourot/he normal"
        image_feature = tf.keras.layers.Conv2D(n_filters, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(x)
        # Conv2D --> ReLU : Initializtion "Glourot/he normal"
        image_feature = tf.keras.layers.Conv2D(n_filters, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(image_feature)
        return image_feature

    def downsample_features(x):
        image_downsampled = tf.keras.layers.MaxPool2D(2)(x)
        image_downsampled = tf.keras.layers.Dropout(0.3)(image_downsampled)
        return image_downsampled

    def double_conv_downsample(x, n_filters):
        to_pass_forward = double_convolution(x, n_filters)
        to_pass_down = downsample_features(to_pass_forward)
        return to_pass_forward, to_pass_down

    # Attention Net
    def attention_gate(from_left, from_bottom, n_filters):
        # Upsample the bottom feature maps to match the left feature maps
        from_bottom = tf.keras.layers.Conv2DTranspose(n_filters, 2, strides=(2, 2), padding='same')(from_bottom)

        g = tf.keras.layers.Conv2D(n_filters, 1, padding = 'same')(from_bottom)
        g = tf.keras.layers.BatchNormalization()(g)

        x = tf.keras.layers.Conv2D(n_filters, 1, padding = 'same')(from_left)
        x = tf.keras.layers.BatchNormalization()(x)

        # Add -> ReLU -> Conv2D -> Sigmoid
        attention_output = tf.keras.layers.Activation('relu')(g + x)
        attention_output = tf.keras.layers.Conv2D(n_filters, 1, padding = 'same')(attention_output)
        attention_output = tf.keras.layers.Activation('sigmoid')(attention_output)
        attention_output = tf.keras.layers.multiply([attention_output, g])
        return attention_output

    # Updampling
    def upsample_concatenate(x, conv_features, n_filters):
        # Upsample the incoming feature
        upsampled = tf.keras.layers.Conv2DTranspose(n_filters, 3, 2, padding = 'same')(x)
        g = attention_gate(conv_features, x, n_filters)
        # Concatenate the features
        concatenated_features = tf.keras.layers.concatenate([upsampled, g])
        # Dropout
        concatenated_features = tf.keras.layers.Dropout(0.3)(concatenated_features)
        # Double COnv
        output = double_convolution(concatenated_features, n_filters)

        return output

    # Inputs
    inputs = tf.keras.layers.Input(shape = image_shape)

    # Encoder for Down Sampling
    # Start Stacking
    forward_1, pass_down_1 = double_conv_downsample(inputs, n_filters = 64)
    forward_2, pass_down_2 = double_conv_downsample(pass_down_1, n_filters = 128)
    forward_3, pass_down_3 = double_conv_downsample(pass_down_2, n_filters = 256)
    forward_4, pass_down_4 = double_conv_downsample(pass_down_3, n_filters = 512)

    # Bottleneck
    Embedding = double_convolution(pass_down_4, n_filters = 1024)

    # Upsample Process
    upsample_1 = upsample_concatenate(Embedding, forward_4, n_filters = 512)
    upsample_2 = upsample_concatenate(upsample_1, forward_3, n_filters = 256)
    upsample_3 = upsample_concatenate(upsample_2, forward_2, n_filters = 128)
    upsample_4 = upsample_concatenate(upsample_3, forward_1, n_filters = 64)

    # Final Output
    outputs = tf.keras.layers.Conv2D(1, 1, padding = 'same', activation = 'sigmoid', kernel_initializer = 'he_normal')(upsample_4)

    return tf.keras.Model(inputs, outputs, name = 'Attention-UNET')