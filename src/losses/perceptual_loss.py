import tensorflow as tf
from keras.losses import LossFunctionWrapper, losses_utils
from tensorflow.keras import models
from tensorflow.keras.applications import VGG19


class VGGPerceptualLoss(LossFunctionWrapper):

    def __init__(
        self,
        input_size,
        vgg_space_loss,
        content_layer,
        image_space_loss,
        vgg_weight,
        perceptual_model="vgg19-imagenet",
        weights_path=None,
        reduction=losses_utils.ReductionV2.AUTO,
        name="vgg_perceptual_error",
    ):
        if perceptual_model == "vgg19-imagenet":
            vgg = VGG19(
                include_top=False,
                weights="imagenet",
                input_shape=[int(input_size), int(input_size), 3],
            )
        elif perceptual_model == "autoencoder":
            vgg = encoder(
                tf.keras.layers.Input(shape=(int(input_size), int(input_size), 3))
            )
            vgg.load_weights(weights_path).expect_partial()
        elif perceptual_model == "vgg19-custom":
            vgg = VGG19(
                include_top=False,
                weights=None,
                input_shape=[int(input_size), int(input_size), 3],
            )
            vgg.load_weights(weights_path).expect_partial()
        else:
            raise Exception(f"Invalid perceptual model: {perceptual_model}")

        vgg.trainable = False
        model = models.Model(
            [vgg.input], vgg.get_layer(content_layer).output, name="vggL"
        )

        # Store loss components
        self.image_loss_value = None
        self.perceptual_loss_value = None

        super().__init__(
            self.loss_fn,
            name=name,
            reduction=reduction,
            model=model,
            vgg_space_loss=vgg_space_loss,
            image_space_loss=image_space_loss,
            vgg_weight=vgg_weight,
        )

    def loss_fn(
        self, y_true, y_pred, model, vgg_space_loss, image_space_loss, vgg_weight
    ):
        losses = vgg_error(
            y_true, y_pred, model, vgg_space_loss, image_space_loss, vgg_weight
        )

        # Store components for later access
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


def vgg_error(y_true, y_pred, model, vgg_space_loss, image_space_loss, vgg_weight):
    vgg_true = model(tensor_preprocessing(y_true))
    vgg_pred = model(tensor_preprocessing(y_pred))

    image_loss = 0.0
    if image_space_loss == "mse":
        image_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    elif image_space_loss == "mae":
        image_loss = tf.keras.losses.mean_absolute_error(y_true, y_pred)

    perceptual = 0.0
    if vgg_space_loss == "mse":
        perceptual = tf.keras.losses.mean_squared_error(vgg_true, vgg_pred)
    elif vgg_space_loss == "mae":
        perceptual = tf.keras.losses.mean_absolute_error(vgg_true, vgg_pred)

    image_loss = tf.reduce_mean(tf.cast(image_loss, tf.float32))
    perceptual = tf.reduce_mean(tf.cast(perceptual, tf.float32))

    total_loss = image_loss + (vgg_weight * perceptual)

    return {
        "total_loss": total_loss,
        "image_loss": image_loss,
        "perceptual": perceptual,
    }


def tensor_preprocessing(data):
    return tf.repeat(data, 3, -1) * 255


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


if __name__ == "__main__":
    import numpy as np

    # Load sample data
    dir_pred = "/home/gamma/Datasets/Mayo-Challenge/npy_img/3mm B30/L067_0_input.npy"
    dir_truth = "/home/gamma/Datasets/Mayo-Challenge/npy_img/3mm B30/L067_0_target.npy"

    img_pred = np.expand_dims(np.expand_dims(np.load(dir_pred), axis=-1), axis=0) / 255
    img_truth = (
        np.expand_dims(np.expand_dims(np.load(dir_truth), axis=-1), axis=0) / 255
    )
    print(f"{img_pred.shape}\n{img_truth.shape}")

    # Initialize loss
    vgg_loss = VGGPerceptualLoss(img_pred.shape[1], "mse", "block5_conv2", "mse", 0.1)

    # Compute loss
    result = vgg_loss(img_truth, img_pred)

    # Get loss components
    components = vgg_loss.get_loss_components()

    print(f"Total Loss: {result.numpy()}")
    print(f"Image Loss: {components['image_loss']}")
    print(f"Perceptual Loss: {components['perceptual']}")
