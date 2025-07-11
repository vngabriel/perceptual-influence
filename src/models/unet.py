import tensorflow as tf


class DoubleConv(tf.keras.models.Model):
    def __init__(
        self,
        filters,
        act_fn="relu",
        use_batchnorm=False,
        ini=None,
        ini_bias=None,
        seed=None,
    ):
        super(DoubleConv, self).__init__()

        if ini == None:
            ini = tf.keras.initializers.GlorotUniform(seed=seed)
        if ini_bias == None:
            ini_bias = tf.keras.initializers.Zeros()

        use_bias = not use_batchnorm
        self.use_batchnorm = use_batchnorm

        self.conv0 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            activation=None,
            use_bias=use_bias,
            padding="same",
            kernel_initializer=ini,
            bias_initializer=ini_bias,
        )
        self.conv1 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            activation=None,
            use_bias=use_bias,
            padding="same",
            kernel_initializer=ini,
            bias_initializer=ini_bias,
        )

        self.batchnorm0, self.batchnorm1 = None, None
        if use_batchnorm:
            self.batchnorm0 = tf.keras.layers.BatchNormalization()
            self.batchnorm1 = tf.keras.layers.BatchNormalization()

        assert act_fn.lower() in [None, "relu", "lrelu", "leaky_relu"]
        self.act_fn1 = act_fn.lower() if act_fn != None else None
        self.act_fn2 = self.act_fn1
        if self.act_fn1 == "relu":
            self.act_fn1 = tf.nn.relu
            self.act_fn2 = self.act_fn1
        elif self.act_fn1 in ["lrelu", "leaky_relu"]:
            self.act_fn1 = tf.nn.leaky_relu
            self.act_fn2 = self.act_fn1

    def call(self, inputs, training=False):
        x = self.conv0(inputs)
        if self.use_batchnorm:
            x = self.batchnorm0(x, training=training)
        if self.act_fn1 != None:
            x = self.act_fn1(x)

        x = self.conv1(x)
        if self.use_batchnorm:
            x = self.batchnorm1(x, training=training)
        if self.act_fn1 != None:
            x = self.act_fn2(x)
        return x


# arXiv:1505.04597
class UNET(tf.keras.models.Model):
    def __init__(
        self, output_actv="sigmoid", channel=64, seed=None, use_batchnorm=False
    ):
        super(UNET, self).__init__()

        # ini = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=seed)
        # ini_bias = ini

        # Default tensorflow initializer
        ini = tf.keras.initializers.GlorotUniform(seed=seed)
        ini_bias = tf.keras.initializers.Zeros()

        div = 64 / channel
        self.block0_conv = DoubleConv(
            filters=64 // div,
            act_fn="relu",
            use_batchnorm=use_batchnorm,
            ini=ini,
            ini_bias=ini_bias,
            seed=seed,
        )

        self.block1_conv = DoubleConv(
            filters=128 // div,
            act_fn="relu",
            use_batchnorm=use_batchnorm,
            ini=ini,
            ini_bias=ini_bias,
            seed=seed,
        )

        self.block2_conv = DoubleConv(
            filters=256 // div,
            act_fn="relu",
            use_batchnorm=use_batchnorm,
            ini=ini,
            ini_bias=ini_bias,
            seed=seed,
        )

        self.block3_conv = DoubleConv(
            filters=512 // div,
            act_fn="relu",
            use_batchnorm=use_batchnorm,
            ini=ini,
            ini_bias=ini_bias,
            seed=seed,
        )

        self.block4_conv = DoubleConv(
            filters=1024 // div,
            act_fn="relu",
            use_batchnorm=use_batchnorm,
            ini=ini,
            ini_bias=ini_bias,
            seed=seed,
        )
        self.block4_deconv0 = tf.keras.layers.Conv2DTranspose(
            filters=512 // div,
            kernel_size=(2, 2),
            activation=None,
            use_bias=True,
            strides=(2, 2),
            padding="same",
            kernel_initializer=ini,
            bias_initializer=ini_bias,
        )

        self.block5_conv = DoubleConv(
            filters=512 // div,
            act_fn="relu",
            use_batchnorm=use_batchnorm,
            ini=ini,
            ini_bias=ini_bias,
            seed=seed,
        )
        self.block5_deconv0 = tf.keras.layers.Conv2DTranspose(
            filters=256 // div,
            kernel_size=(2, 2),
            activation=None,
            use_bias=True,
            strides=(2, 2),
            padding="same",
            kernel_initializer=ini,
            bias_initializer=ini_bias,
        )

        self.block6_conv = DoubleConv(
            filters=256 // div,
            act_fn="relu",
            use_batchnorm=use_batchnorm,
            ini=ini,
            ini_bias=ini_bias,
            seed=seed,
        )
        self.block6_deconv0 = tf.keras.layers.Conv2DTranspose(
            filters=128 // div,
            kernel_size=(2, 2),
            activation=None,
            use_bias=True,
            strides=(2, 2),
            padding="same",
            kernel_initializer=ini,
            bias_initializer=ini_bias,
        )

        self.block7_conv = DoubleConv(
            filters=128 // div,
            act_fn="relu",
            use_batchnorm=use_batchnorm,
            ini=ini,
            ini_bias=ini_bias,
            seed=seed,
        )
        self.block7_deconv0 = tf.keras.layers.Conv2DTranspose(
            filters=64 // div,
            kernel_size=(2, 2),
            activation=None,
            use_bias=True,
            strides=(2, 2),
            padding="same",
            kernel_initializer=ini,
            bias_initializer=ini_bias,
        )

        self.block8_conv = DoubleConv(
            filters=64 // div,
            act_fn="relu",
            use_batchnorm=use_batchnorm,
            ini=ini,
            ini_bias=ini_bias,
            seed=seed,
        )

        self.final_conv = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(1, 1),
            padding="same",
            kernel_initializer=ini,
            bias_initializer=ini_bias,
        )

        self.output_actv = output_actv.lower()

        self.MaxPooling2D = tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2), strides=None, padding="valid"
        )
        self.Concatenate = tf.keras.layers.Concatenate()

    def call(self, inputs, training=None):

        # BLOCK 0
        x0 = self.block0_conv(inputs, training=training)
        x = self.MaxPooling2D(x0)

        # BLOCK 1
        x1 = self.block1_conv(x, training=training)
        x = self.MaxPooling2D(x1)

        # BLOCK 2
        x2 = self.block2_conv(x, training=training)
        x = self.MaxPooling2D(x2)

        # BLOCK 3
        x3 = self.block3_conv(x, training=training)
        x = self.MaxPooling2D(x3)

        # BLOCK 4
        x = self.block4_conv(x, training=training)
        x = self.block4_deconv0(x, training=training)

        x = self.Concatenate([x, x3])

        # BLOCK 5
        x = self.block5_conv(x, training=training)
        x = self.block5_deconv0(x, training=training)

        x = self.Concatenate([x, x2])

        # BLOCK 6
        x = self.block6_conv(x, training=training)
        x = self.block6_deconv0(x, training=training)

        x = self.Concatenate([x, x1])

        # BLOCK 7
        x = self.block7_conv(x, training=training)
        x = self.block7_deconv0(x, training=training)

        x = self.Concatenate([x, x0])

        # BLOCK 8
        x = self.block8_conv(x, training=training)
        x = self.final_conv(x, training=training)  # Conv 1x1x1

        if self.output_actv == "sigmoid":
            x = tf.nn.sigmoid(x)
        elif self.output_actv == "tanh":
            x = tf.nn.tanh(x)
        return x


if __name__ == "__main__":
    import numpy as np
    import tensorflow.keras.backend as K

    model = UNET(seed=10)
    pred = model(np.random.rand(1, 512, 512, 1), training=True)
    print(pred.shape)
    model.summary()

    # Calculates the number of trainable weights
    print(np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_weights]))

    trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
    print(trainable_count)

    print(model.trainable_weights[0][0][0][0][0])
