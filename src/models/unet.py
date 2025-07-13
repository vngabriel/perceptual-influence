import tensorflow as tf


class DoubleConv(tf.keras.models.Model):
    def __init__(
        self,
        filters,
        activation_function="relu",
        use_batch_normalization=False,
        ini=None,
        ini_bias=None,
        seed=None,
    ):
        super(DoubleConv, self).__init__()

        if ini is None:
            ini = tf.keras.initializers.GlorotUniform(seed=seed)
        if ini_bias is None:
            ini_bias = tf.keras.initializers.Zeros()

        use_bias = not use_batch_normalization
        self.use_batch_normalization = use_batch_normalization

        self.conv_0 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            activation=None,
            use_bias=use_bias,
            padding="same",
            kernel_initializer=ini,
            bias_initializer=ini_bias,
        )
        self.conv_1 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            activation=None,
            use_bias=use_bias,
            padding="same",
            kernel_initializer=ini,
            bias_initializer=ini_bias,
        )

        self.batch_normalization_0 = None
        self.batch_normalization_1 = None
        if use_batch_normalization:
            self.batch_normalization_0 = tf.keras.layers.BatchNormalization()
            self.batch_normalization_1 = tf.keras.layers.BatchNormalization()

        assert activation_function.lower() in [None, "relu", "lrelu", "leaky_relu"]

        self.activation_function_1 = None
        self.activation_function_2 = None
        if activation_function.lower() == "relu":
            self.activation_function_1 = tf.nn.relu
            self.activation_function_2 = self.activation_function_1

        elif activation_function.lower() in ["lrelu", "leaky_relu"]:
            self.activation_function_1 = tf.nn.leaky_relu
            self.activation_function_2 = self.activation_function_1

    def call(self, inputs, training=False):
        x = self.conv_0(inputs)
        if self.use_batch_normalization:
            x = self.batch_normalization_0(x, training=training)
        if self.activation_function_1 is not None:
            x = self.activation_function_1(x)

        x = self.conv_1(x)
        if self.use_batch_normalization:
            x = self.batch_normalization_1(x, training=training)
        if self.activation_function_1 is not None:
            x = self.activation_function_2(x)

        return x


class Unet(tf.keras.models.Model):
    def __init__(
        self,
        output_activation="sigmoid",
        channel=64,
        seed=None,
        use_batch_normalization=False,
    ):
        super(Unet, self).__init__()

        ini = tf.keras.initializers.GlorotUniform(seed=seed)
        ini_bias = tf.keras.initializers.Zeros()

        div = 64 / channel
        self.block0_conv = DoubleConv(
            filters=64 // div,
            activation_function="relu",
            use_batch_normalization=use_batch_normalization,
            ini=ini,
            ini_bias=ini_bias,
            seed=seed,
        )
        self.block1_conv = DoubleConv(
            filters=128 // div,
            activation_function="relu",
            use_batch_normalization=use_batch_normalization,
            ini=ini,
            ini_bias=ini_bias,
            seed=seed,
        )
        self.block2_conv = DoubleConv(
            filters=256 // div,
            activation_function="relu",
            use_batch_normalization=use_batch_normalization,
            ini=ini,
            ini_bias=ini_bias,
            seed=seed,
        )
        self.block3_conv = DoubleConv(
            filters=512 // div,
            activation_function="relu",
            use_batch_normalization=use_batch_normalization,
            ini=ini,
            ini_bias=ini_bias,
            seed=seed,
        )
        self.block4_conv = DoubleConv(
            filters=1024 // div,
            activation_function="relu",
            use_batch_normalization=use_batch_normalization,
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
            activation_function="relu",
            use_batch_normalization=use_batch_normalization,
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
            activation_function="relu",
            use_batch_normalization=use_batch_normalization,
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
            activation_function="relu",
            use_batch_normalization=use_batch_normalization,
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
            activation_function="relu",
            use_batch_normalization=use_batch_normalization,
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

        self.output_activation = output_activation.lower()

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
        x = self.final_conv(x, training=training)

        if self.output_activation == "sigmoid":
            x = tf.nn.sigmoid(x)
        elif self.output_activation == "tanh":
            x = tf.nn.tanh(x)

        return x
