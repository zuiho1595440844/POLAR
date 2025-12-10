from tensorflow.keras.layers import Input, Conv2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import DepthwiseConv2D, GlobalAveragePooling2D


def modified_vgg16_model(input_shape, l2_regularizer=0.001):
    reg = regularizers.l2(l2_regularizer)  # 定义L2正则化项

    input_layer = Input(shape=input_shape)
    print(input_shape)
    # Block 1
    x = ZeroPadding2D(padding=(1, 1))(input_layer)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 2
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 3
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 4
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 5
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Flatten and dense layers
    x = Flatten()(x)
    x = Dense(4096, activation='relu', kernel_regularizer=reg)(x)
    x = Dropout(0.5)(x)

    # Output layers
    policy_output = Dense(19 * 19 + 1, activation='softmax', name='policy_output', kernel_regularizer=reg)(x)
    value_output = Dense(1, name='value_output', kernel_regularizer=reg)(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=[policy_output, value_output])

    return model


def lightweight_vgg16_model(input_shape, l2_regularizer=0.001):
    reg = regularizers.l2(l2_regularizer)

    input_layer = Input(shape=input_shape)

    # Block 1
    x = ZeroPadding2D(padding=(1, 1))(input_layer)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 2
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = DepthwiseConv2D((3, 3), activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = Conv2D(64, (1, 1), activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 3
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = DepthwiseConv2D((3, 3), activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 4
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = DepthwiseConv2D((3, 3), activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = Conv2D(256, (1, 1), activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 5
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = DepthwiseConv2D((3, 3), activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = Conv2D(512, (1, 1), activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Flatten and dense layers
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=reg)(x)
    x = Dropout(0.5)(x)

    # Output layers
    policy_output = Dense(19 * 19 + 1, activation='softmax', name='policy_output', kernel_regularizer=reg)(x)
    value_output = Dense(1, name='value_output', kernel_regularizer=reg)(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=[policy_output, value_output])

    return model


def lightweight_vgg16_model_op(input_shape):
    input_layer = Input(shape=input_shape)

    # Block 1
    x = ZeroPadding2D(padding=(1, 1))(input_layer)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 2
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = DepthwiseConv2D((3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Conv2D(64, (1, 1), padding='same')(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 3
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = DepthwiseConv2D((3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Conv2D(128, (1, 1), padding='same')(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 4
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = DepthwiseConv2D((3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Conv2D(256, (1, 1), padding='same')(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 5
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = DepthwiseConv2D((3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Conv2D(512, (1, 1), padding='same')(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Flatten and dense layers
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)  # Reduced size
    x = Dropout(0.5)(x)

    # Output layers
    policy_output = Dense(19 * 19, activation='softmax', name='policy_output')(x)
    value_output = Dense(1, name='value_output')(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=[policy_output, value_output])

    return model


import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_addons as tfa


def DWConv_TF(filters):
    return models.Sequential([
        layers.DepthwiseConv2D(kernel_size=3, padding='same'),
        layers.LeakyReLU(0.01),
        layers.Conv2D(filters, kernel_size=1, padding='same'),
        layers.LeakyReLU(0.01)
    ])


def UCMBlock_TF(filters, mlp_ratio=4):
    hidden_dim = filters * mlp_ratio
    return models.Sequential([
        layers.LayerNormalization(),
        layers.Conv2D(hidden_dim, 1),
        layers.DepthwiseConv2D(3, padding='same', depth_multiplier=1),
        layers.LeakyReLU(0.01),
        layers.Conv2D(filters, 1),
        layers.Dropout(0.1)
    ])


def OverlapPatchEmbed_TF(filters, patch_size=3, stride=2):
    return models.Sequential([
        layers.Conv2D(filters, kernel_size=patch_size, strides=stride, padding='same'),
        layers.LayerNormalization()
    ])


def UCM_GoNet_TF(input_shape=(19, 19, 3)):
    inputs = layers.Input(shape=input_shape)

    # Initial conv
    x = layers.Conv2D(8, 3, padding='same')(inputs)
    x = tfa.layers.GroupNormalization(groups=4)(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    # Patch + UCM blocks
    patch1 = OverlapPatchEmbed_TF(16)(x)
    x = UCMBlock_TF(16)(patch1)

    patch2 = OverlapPatchEmbed_TF(24)(x)
    x = UCMBlock_TF(24)(patch2)

    patch3 = OverlapPatchEmbed_TF(32)(x)
    x = UCMBlock_TF(32)(patch3)

    patch4 = OverlapPatchEmbed_TF(48)(x)
    x = UCMBlock_TF(48)(patch4)

    # Global pooling
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Output heads
    policy_out = layers.Dense(19 * 19, activation='softmax', name='policy_output')(x)
    value_out = layers.Dense(1, name='value_output')(x)

    model = models.Model(inputs=inputs, outputs=[policy_out, value_out])
    return model
