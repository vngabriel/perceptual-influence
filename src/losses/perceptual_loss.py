import tensorflow as tf
from keras.losses import LossFunctionWrapper, losses_utils
from tensorflow.keras import models
from tensorflow.keras.applications import VGG19


class PerceptualLoss(LossFunctionWrapper):
    def __init__(
        self,
        input_size,
        perceptual_space_loss,
        content_layer,
        image_space_loss,
        perceptual_weight,
        perceptual_model,
        weights_path,
        reduction=losses_utils.ReductionV2.AUTO,
        name="perceptual_error",
    ):
        if perceptual_model == "vgg19-imagenet":
            encoder_model = VGG19(
                include_top=False,
                weights="imagenet",
                input_shape=[int(input_size), int(input_size), 3],
            )
        elif perceptual_model == "autoencoder":
            encoder_model = encoder(
                tf.keras.layers.Input(shape=(int(input_size), int(input_size), 3))
            )
            encoder_model.load_weights(weights_path).expect_partial()
        elif perceptual_model == "vgg19-custom":
            encoder_model = VGG19(
                include_top=False,
                weights=None,
                input_shape=[int(input_size), int(input_size), 3],
            )
            encoder_model.load_weights(weights_path).expect_partial()
        else:
            raise Exception(f"Invalid perceptual model: {perceptual_model}")

        encoder_model.trainable = False
        model = models.Model(
            [encoder_model.input],
            encoder_model.get_layer(content_layer).output,
            name="perceptual_model",
        )

        self.image_loss_value = None
        self.perceptual_loss_value = None

        super().__init__(
            self.loss_function,
            name=name,
            reduction=reduction,
            model=model,
            perceptual_space_loss=perceptual_space_loss,
            image_space_loss=image_space_loss,
            perceptual_weight=perceptual_weight,
        )

    def loss_function(
        self,
        y_true,
        y_pred,
        model,
        perceptual_space_loss,
        image_space_loss,
        perceptual_weight,
    ):
        losses = perceptual_error(
            y_true,
            y_pred,
            model,
            perceptual_space_loss,
            image_space_loss,
            perceptual_weight,
        )

        self.image_loss_value = losses["image_loss"]
        self.perceptual_loss_value = losses["perceptual"]

        return losses["total_loss"]

    def get_loss_components(self):
        components = {
            "image_loss": (
                self.image_loss_value if self.image_loss_value is not None else None
            ),
            "perceptual": (
                self.perceptual_loss_value
                if self.perceptual_loss_value is not None
                else None
            ),
        }

        self.image_loss_value = None
        self.perceptual_loss_value = None

        return components


def encoder(inputs):
    x = tf.keras.layers.Conv2D(
        64, (3, 3), padding="same", activation="relu", name="block1_conv1"
    )(inputs)
    x = tf.keras.layers.Conv2D(
        64, (3, 3), padding="same", activation="relu", name="block1_conv2"
    )(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.Conv2D(
        128, (3, 3), padding="same", activation="relu", name="block2_conv1"
    )(x)
    x = tf.keras.layers.Conv2D(
        128, (3, 3), padding="same", activation="relu", name="block2_conv2"
    )(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.Conv2D(
        256, (3, 3), padding="same", activation="relu", name="block3_conv1"
    )(x)
    x = tf.keras.layers.Conv2D(
        256, (3, 3), padding="same", activation="relu", name="block3_conv2"
    )(x)
    x = tf.keras.layers.Conv2D(
        256, (3, 3), padding="same", activation="relu", name="block3_conv3"
    )(x)
    y = tf.keras.layers.Conv2D(
        256, (3, 3), padding="same", activation="relu", name="block3_conv4"
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=y, name="SACNN-Encoder")

    return model


def perceptual_error(
    y_true, y_pred, model, perceptual_space_loss, image_space_loss, perceptual_weight
):
    perceptual_true = model(tensor_preprocessing(y_true))
    perceptual_pred = model(tensor_preprocessing(y_pred))

    image_loss = 0.0
    if image_space_loss == "mse":
        image_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    elif image_space_loss == "mae":
        image_loss = tf.keras.losses.mean_absolute_error(y_true, y_pred)

    perceptual = 0.0
    if perceptual_space_loss == "mse":
        perceptual = tf.keras.losses.mean_squared_error(
            perceptual_true, perceptual_pred
        )
    elif perceptual_space_loss == "mae":
        perceptual = tf.keras.losses.mean_absolute_error(
            perceptual_true, perceptual_pred
        )

    image_loss = tf.reduce_mean(tf.cast(image_loss, tf.float32))
    perceptual = tf.reduce_mean(tf.cast(perceptual, tf.float32))

    total_loss = image_loss + (perceptual_weight * perceptual)

    return {
        "total_loss": total_loss,
        "image_loss": image_loss,
        "perceptual": perceptual,
    }


def tensor_preprocessing(data):
    return tf.repeat(data, 3, -1) * 255
