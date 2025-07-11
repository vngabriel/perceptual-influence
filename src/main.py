import configparser
import datetime
import json
import os
import random
import time

import imageio
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
from plotly.offline import plot as save_plot
from plotly.subplots import make_subplots
from skimage.metrics import (
    structural_similarity,
    peak_signal_noise_ratio,
    normalized_root_mse,
)
from tensorflow.keras.utils import Progbar

from src.data.mayo_challenge import MayoChallenge_DataGenerator
from src.losses.perceptual_loss import VGGPerceptualLoss
from src.models.unet import UNET
from src.utils.directory_tools import create_folder
from src.utils.system_monitor import SystemMonitor

# To make execution deterministic (will make execution slower)
os.environ["TF_DETERMINISTIC_OPS"] = "1"


class Experiment:

    def __init__(self, init_file, specific_hash=None):
        print(f"\n\t >> EXECUTING {init_file} CONFIG FILE <<\n")

        self.train_gen = None
        self.val_gen = None
        self.test_gen = None
        self.model = None
        self.optimizer = None
        self.loss = None
        self.train_acc = None
        self.val_acc = None
        self.seed = None
        self.train_setup = None
        self.test_setup = None
        self.network_setup = None
        self.loss_setup = None
        self.hash = specific_hash
        self.sys_monitor = SystemMonitor()
        self.config = configparser.ConfigParser()
        self.config.read(init_file)
        self.management = self.config["Experiment-Management"]

        random_hash = random.getrandbits(32)

        create_folder("./outputs")
        create_folder("./weights/")
        if (
            self.config["Experiment-Management"]["mode"].lower() == "test"
            or self.management["resume-previous-exp"].lower() == "yes"
        ):

            self.hash = self.management["previous-hash"]
            print(
                "LOADING WEIGHTS FROM THE PREVIOUS EXECUTION {}".format(str(self.hash))
            )
            with open("./outputs/train-setup.{}".format(str(self.hash))) as json_file:
                self.train_setup = json.load(json_file)
                if self.train_setup["seed"].lower() not in ["none", "", " "]:
                    self.seed = int(self.train_setup["seed"])
                    self.__set_experiment_seed()

            if os.path.exists(f"./outputs/loss-setup.{str(self.hash)}"):
                with open(
                    "./outputs/loss-setup.{}".format(str(self.hash))
                ) as json_file:
                    self.loss_setup = json.load(json_file)

            with open("./outputs/network-setup.{}".format(str(self.hash))) as json_file:
                self.network_setup = json.load(json_file)

        else:
            if self.hash == None:
                self.hash = random_hash

            print("NEW EXPERIMENT HASH: {}".format(str(self.hash)))

            if self.management["mode"].lower() == "train":
                print("STARTING TRAINING MODE")
                # train parameters may come from a CURRENT config file
                self.train_setup = self.config["Train-Setup"]
                if self.train_setup["seed"].lower() not in ["none", "", " "]:
                    self.seed = int(self.train_setup["seed"])
                    self.__set_experiment_seed()

                self.network_setup = self.config["Network-Hyperparameters"]

                if self.train_setup["loss"].lower() in ["vgg"]:
                    self.loss_setup = self.config[
                        f"{self.train_setup['loss'].upper()}-Loss-Hyperparameters"
                    ]

                with open(
                    "./outputs/train-setup.{}".format(str(self.hash)), "w"
                ) as outfile:
                    json.dump(dict(self.train_setup), outfile, indent=2)

                with open(
                    "./outputs/network-setup.{}".format(str(self.hash)), "w"
                ) as outfile:
                    json.dump(dict(self.network_setup), outfile, indent=2)

                if self.loss_setup is not None:
                    with open(
                        "./outputs/loss-setup.{}".format(str(self.hash)), "w"
                    ) as outfile:
                        json.dump(dict(self.loss_setup), outfile, indent=2)
            else:
                print("\nNOTHING TO DO. IS NOT POSSIBLE TEST A UNTRAINED MODEL!")
                print("'resume-previous-exp' is equals to 'yes'.")
                exit()

    def __set_experiment_seed(self):
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

    def execute(self):
        if self.management["mode"].lower() == "train":
            self.__train()
        else:
            self.__test()

    def load_data(self):
        if self.train_setup["database"].lower() == "mayo-challenge":
            with open(
                "./src/data/mayo-challenge-split.{}".format(
                    self.train_setup["split-hash"]
                )
            ) as json_file:
                data = json.load(json_file)

            negative_normalize = self.train_setup["negative_values"].lower() == "yes"
            if self.management["mode"].lower() == "train":
                self.train_gen = MayoChallenge_DataGenerator(
                    data_src=self.management["data-dir"],
                    scans=data["train"],
                    batch_size=int(self.train_setup["batch"]),
                    patch_size=int(self.train_setup["patch-size"]),
                    patch_stride=int(self.train_setup["patch-skip"]),
                    normalize_0_1=False,
                    negative_normalize=negative_normalize,
                    seed=self.seed,
                )
                self.val_gen = MayoChallenge_DataGenerator(
                    data_src=self.management["data-dir"],
                    scans=data["validation"],
                    batch_size=int(self.train_setup["batch"]),
                    patch_size=int(self.train_setup["patch-size"]),
                    patch_stride=int(self.train_setup["patch-skip"]),
                    normalize_0_1=False,
                    negative_normalize=negative_normalize,
                    seed=self.seed,
                )
            else:
                self.val_gen = MayoChallenge_DataGenerator(
                    data_src=self.management["data-dir"],
                    scans=data["validation"],
                    batch_size=int(self.train_setup["batch"]),
                    patch_size=int(self.train_setup["patch-size"]),
                    patch_stride=int(self.train_setup["patch-skip"]),
                    normalize_0_1=False,
                    negative_normalize=negative_normalize,
                    seed=self.seed,
                )
                self.test_gen = MayoChallenge_DataGenerator(
                    data_src=self.management["data-dir"],
                    scans=data["test"],
                    batch_size=int(self.train_setup["batch"]),
                    patch_size=int(self.train_setup["patch-size"]),
                    patch_stride=int(self.train_setup["patch-skip"]),
                    normalize_0_1=False,
                    negative_normalize=negative_normalize,
                    seed=self.seed,
                )
        else:
            raise Exception("Database not available")

    def load_model(self):
        """
        Load the selected network and loss function to be used in the current experiment.
        """
        negative_output = self.train_setup["negative_values"].lower() == "yes"
        if self.train_setup["network"].lower() == "unet":
            self.model = UNET(
                output_actv="tanh" if negative_output else "sigmoid",
                channel=int(self.network_setup["channels"]),
                seed=self.seed,
            )
        else:
            raise Exception(
                "Network not available: {}".format(self.train_setup["network"])
            )

        _ = self.model(
            np.zeros(self.val_gen.__getitem__(0)[0].shape), training=True
        )  # build

        self.model.summary()

        if self.train_setup["loss"].lower() == "mse":
            self.loss = tf.keras.losses.MeanSquaredError()
        elif self.train_setup["loss"].lower() == "mae":
            self.loss = tf.keras.losses.MeanAbsoluteError()
        elif self.train_setup["loss"].lower() == "vgg":
            print(f"Using VGG Weight: {float(self.loss_setup['vgg-weight'])}")
            print(f"Using Content Layer: {self.loss_setup['vgg-content-layer']}")
            print(f"Perceptual model: {self.loss_setup['perceptual-model']}")
            print(f"Weights path: {self.loss_setup['weights-path']}")
            self.loss = VGGPerceptualLoss(
                input_size=self.train_setup["patch-size"],
                vgg_space_loss=self.loss_setup["vgg-space-loss"],
                image_space_loss=self.loss_setup["image-space-loss"],
                vgg_weight=float(self.loss_setup["vgg-weight"]),
                content_layer=self.loss_setup["vgg-content-layer"],
                perceptual_model=self.loss_setup["perceptual-model"],
                weights_path=self.loss_setup["weights-path"],
            )
        else:
            raise Exception("Loss Function not available")

    @tf.function
    def __train_step(self, x, y):
        """
        Private method that computes one step of the Backpropagation technique
            to update the network weights during the
            training stage.

        :param x: 4D tensor - [batch,height,width,channel] - composed of
            low-dose CT images - i.e. network input - from the training set.
        :type x: :class:`tf.tensor`
        :param y: 4D tensor - [batch,height,width,channel] - composed of the
            high-dose CT images - i.e. target output - relative to the data in
            x. training set.
        :type y: :class:`tf.tensor`

        """
        with tf.GradientTape() as tape:
            pred = self.model(x, training=True)
            err = self.loss(y, pred)

        if self.train_setup["loss"].lower() == "vgg":
            components = self.loss.get_loss_components()
        else:
            components = {}

        grads = tape.gradient(err, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.train_acc.update_state(y, pred)

        return err, components

    @tf.function
    def __val_step(self, x, y):
        """
        Private method that computes the network performance for images into a
        validation set. Tracking the accuracy levels into the validation dataset
        is an useful technique to avoid overfitting.

        :param x: 4D tensor - [batch,height,width,channel] - composed of
            low-dose CT images - i.e. network input - from the validation set.
        :type x: :class:`tf.tensor`
        :param y: 4D tensor - [batch,height,width,channel] - composed of the
            high-dose CT images - i.e. target output - relative to the data in
            x.training set.
        :type y: :class:`tf.tensor`

        """
        pred = self.model(x, training=False)
        err = self.loss(y, pred)

        if self.train_setup["loss"].lower() == "vgg":
            components = self.loss.get_loss_components()
        else:
            components = {}

        self.val_acc.update_state(y, pred)

        return err, components

    def __get_trainable_weights(self):
        total = np.sum(
            [np.prod(v.get_shape().as_list()) for v in self.model.trainable_weights]
        )
        return int(total)

    def __train(self):
        """
        Private method that runs the complete training stage composed of the
        number of epochs indicated in the configuration `.ini` file. For each
        epoch, a :func:`__val_step` is executed after each :func:`__train_step`.
        Losses and accuracies for both training and validation sets are saved in
        the end of each epoch in the json file ./outputs/logs/train-hist.<hash>.

        """
        if self.management["mode"].lower() != "train":
            raise Exception("Impossible strating training outside the train mode")

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=float(self.train_setup["lr"])
        )
        self.train_acc = tf.keras.metrics.CosineSimilarity(name="train_acc")
        self.val_acc = tf.keras.metrics.CosineSimilarity(name="val_acc")

        first_epoch = 0
        training_time = 0.0

        if self.management["resume-previous-exp"].lower() == "yes":
            first_epoch = int(self.management["last-epoch-executed"])
            with open(f"./outputs/logs/train-hist.{self.hash}") as json_file:
                train_history = json.load(json_file)

            if (
                self.train_setup["overwrite-weights-file"].lower() == "yes"
                or self.train_setup["early-stop"].lower() == "validation-accuracy"
            ):
                self.model.load_weights(f"./weights/{self.hash}")
            else:
                self.model.load_weights(f"./weights/{self.hash}_{first_epoch}")

            print("RESTARTING {} FROM EPOCH {}".format(str(self.hash), first_epoch + 1))
            training_time = float(train_history["total-training-time"])

        else:
            train_history = {
                "train-loss": [],
                "train-acc": [],
                "val-loss": [],
                "val-acc": [],
                "ram-mem-used": [],
                "gpu-mem-free": [],
                "gpu-mem-used": [],
                "elapsed-time-per-epoch": [],
                "best-val": 0.0,
                "last-epoch-saved": 0,
                "GPU": [],
            }
            if self.train_setup["loss"].lower() == "vgg":
                train_history["train-image-loss"] = []
                train_history["train-perceptual-loss"] = []
                train_history["val-image-loss"] = []
                train_history["val-perceptual-loss"] = []

        for epoch in range(first_epoch, int(self.train_setup["epochs"])):
            print(
                "\nTRAINING: epoch {}/{}".format(
                    epoch + 1, int(self.train_setup["epochs"])
                )
            )

            start = time.time()
            pb_train = Progbar(
                len(self.train_gen) * self.train_gen.batch_size,
                stateful_metrics=["train-loss", "train-acc"],
            )

            if self.train_setup["loss"].lower() == "vgg":
                pb_train = Progbar(
                    len(self.train_gen) * self.train_gen.batch_size,
                    stateful_metrics=[
                        "train-loss",
                        "train-image-loss",
                        "train-perceptual-loss",
                        "train-acc",
                    ],
                )

            total_train_loss = 0.0
            total_train_image_loss = 0.0
            total_train_perceptual_loss = 0.0

            for i, (x_tensor, y_tensor) in enumerate(self.train_gen):
                train_loss, train_components = self.__train_step(x_tensor, y_tensor)
                total_train_loss += train_loss

                if self.train_setup["loss"].lower() == "vgg":
                    total_train_image_loss += train_components["image_loss"]
                    total_train_perceptual_loss += train_components["perceptual"]

                # print("Single step values:")
                # print(f"train_loss: {train_loss}")
                # print(f"image_loss: {train_components['image_loss']}")
                # print(f"perceptual_loss: {train_components['perceptual']}")
                # print(f"vgg_weight: {float(self.loss_setup['vgg-weight'])}")
                # print(f"manual calc: {train_components['image_loss'] + (float(self.loss_setup['vgg-weight']) * train_components['perceptual'])}")

                # if i != 0 and i % 15 == 0:
                if i >= 0:
                    if self.train_setup["loss"].lower() == "vgg":
                        manual_loss = (
                            total_train_image_loss
                            + (
                                float(self.loss_setup["vgg-weight"])
                                * total_train_perceptual_loss
                            )
                        ) / (i + 1)

                        # print(f"\nDebug at step {i}:")
                        # print(f"total_train_loss: {total_train_loss}")
                        # print(f"total_train_loss / (i+1): {total_train_loss / (i + 1)}")
                        # print(f"total_image_loss: {total_train_image_loss}")
                        # print(f"total_image_loss / (i+1): {total_train_image_loss / (i + 1)}")
                        # print(f"total_perceptual_loss: {total_train_perceptual_loss}")
                        # print(f"total_perceptual_loss / (i+1): {total_train_perceptual_loss / (i + 1)}")
                        # print(f"manual calc: {manual_loss}")

                        pb_train.update(
                            i * self.train_gen.batch_size,
                            values=[
                                (
                                    "train-loss",
                                    total_train_loss / (i + 1),
                                ),  # mean of loss
                                ("train-image-loss", total_train_image_loss / (i + 1)),
                                (
                                    "train-perceptual-loss",
                                    total_train_perceptual_loss / (i + 1),
                                ),
                                ("train-acc", self.train_acc.result()),
                            ],
                        )
                    else:
                        pb_train.update(
                            i * self.train_gen.batch_size,
                            values=[
                                (
                                    "train-loss",
                                    total_train_loss / (i + 1),
                                ),  # mean of loss
                                ("train-acc", self.train_acc.result()),
                            ],
                        )
            print("")

            # time.sleep(60*3) # To cool the GPU (use in case restart suddenly)

            print(
                "VALIDATING: epoch {}/{}".format(
                    epoch + 1, int(self.train_setup["epochs"])
                )
            )
            pb_val = Progbar(
                len(self.val_gen) * self.val_gen.batch_size,
                stateful_metrics=["val-loss", "val-acc"],
            )

            if self.train_setup["loss"].lower() == "vgg":
                pb_val = Progbar(
                    len(self.val_gen) * self.val_gen.batch_size,
                    stateful_metrics=[
                        "val-loss",
                        "val-image-loss",
                        "val-perceptual-loss",
                        "val-acc",
                    ],
                )

            total_val_loss = 0.0
            total_val_image_loss = 0.0
            total_val_perceptual_loss = 0.0

            for i, (x_tensor, y_tensor) in enumerate(self.val_gen):
                val_loss, val_components = self.__val_step(x_tensor, y_tensor)
                total_val_loss += val_loss

                if self.train_setup["loss"].lower() == "vgg":
                    total_val_image_loss += val_components["image_loss"]
                    total_val_perceptual_loss += val_components["perceptual"]

                # if i != 0 and i % 15 == 0:
                if i >= 0:
                    if self.train_setup["loss"].lower() == "vgg":
                        pb_val.update(
                            i * self.train_gen.batch_size,
                            values=[
                                ("val-loss", total_val_loss / (i + 1)),  # mean of loss
                                ("val-image-loss", total_val_image_loss / (i + 1)),
                                (
                                    "val-perceptual-loss",
                                    total_val_perceptual_loss / (i + 1),
                                ),
                                ("val-acc", self.val_acc.result()),
                            ],
                        )
                    else:
                        pb_val.update(
                            i * self.val_gen.batch_size,
                            values=[
                                ("val-loss", total_val_loss / (i + 1)),  # mean of loss
                                ("val-acc", self.val_acc.result()),
                            ],
                        )
            print("")  # To finish the Progbar

            if self.train_setup["early-stop"].lower() == "validation-accuracy":
                if self.validation_acc.result() > train_history["best-val"]:
                    train_history["best-val"] = float(self.validation_acc.result())
                    self.model.save_weights("./weights/{}".format(str(self.hash)))
                    print("EARLY STOP - Last epoch saved {}".format(epoch))
            else:
                if self.train_setup["overwrite-weights-file"].lower() == "no":
                    self.model.save_weights(f"./weights/{self.hash}_{epoch + 1}")
                    print(f"Last epoch ({epoch}) saved (weight file was NO overwriten)")
                else:
                    self.model.save_weights("./weights/{}".format(str(self.hash)))
                    print(f"Last epoch ({epoch}) saved (weight file was overwriten)")

            end = time.time()
            epoch_elapsed_time = (end - start) / 3600
            gpu_info = self.sys_monitor.get_gpu_info()
            train_history["GPU"].append(gpu_info["name"])
            train_history["gpu-mem-free"].append(gpu_info["mem_free"])
            train_history["gpu-mem-used"].append(gpu_info["mem_used"])
            train_history["ram-mem-used"].append(self.sys_monitor.get_mem_usage())
            train_history["train-loss"].append(
                float(total_train_loss / len(self.train_gen))
            )
            train_history["train-acc"].append(float(self.train_acc.result()))
            train_history["val-loss"].append(float(total_val_loss / len(self.val_gen)))
            train_history["val-acc"].append(float(self.val_acc.result()))
            train_history["elapsed-time-per-epoch"].append(epoch_elapsed_time)

            if self.train_setup["loss"].lower() == "vgg":
                train_history["train-image-loss"].append(
                    float(total_train_image_loss / len(self.train_gen))
                )
                train_history["train-perceptual-loss"].append(
                    float(total_train_perceptual_loss / len(self.train_gen))
                )
                train_history["val-image-loss"].append(
                    float(total_val_image_loss / len(self.val_gen))
                )
                train_history["val-perceptual-loss"].append(
                    float(total_val_perceptual_loss / len(self.val_gen))
                )

            training_time = training_time + epoch_elapsed_time
            train_history["total-training-time"] = training_time
            train_history["trainable-weights"] = self.__get_trainable_weights()

            create_folder("./outputs/logs/")
            with open(
                "./outputs/logs/train-hist.{}".format(str(self.hash)), "w"
            ) as outfile:
                json.dump(train_history, outfile, indent=2)

            self.train_acc.reset_states()
            self.val_acc.reset_states()

    def __hist_curves(self, hist):
        """
        Private method that plots the loss/accuracy curves selected by the user
        in the configuration `.ini` file within the test mode.

        :param hist: dictionary containing the data saved by the __train
        function in the json file ./outputs/logs/train-hist.<hash>.
        :type hist: :class:`dict`

        """
        print(
            "PLOTTING THE ACCURACY AND/OR THE LOSS CURVES DURING THE PREVIOUS TRAINING..."
        )

        # To save the plot history
        create_folder(self.management["output-dir-imgs"])

        data = pd.DataFrame(hist)
        show_plot = False
        plot_data = []

        if self.test_setup["train-loss-curve"].lower() == "yes":
            plot_data.append("train-loss")
            show_plot = True
        if self.test_setup["val-loss-curve"].lower() == "yes":
            plot_data.append("val-loss")
            show_plot = True
        if self.test_setup["train-acc-curve"].lower() == "yes":
            plot_data.append("train-acc")
            show_plot = True
        if self.test_setup["val-acc-curve"].lower() == "yes":
            plot_data.append("val-acc")
            show_plot = True

        if show_plot == True:
            fig = px.line(data, y=plot_data, title="Train History")
            dir_save_plot = os.path.join(
                self.management["output-dir-imgs"], f"{self.hash}_hist.html"
            )
            print(f"SAVING PLOT OF TRAIN HISTORY IN: {dir_save_plot}")

            save_plot(fig, filename=dir_save_plot)
            fig.show()

        if self.test_setup["save-curves-as-csv"].lower() == "yes":
            csv_file = open(
                "./outputs/train_hist_curves-{}.csv".format(str(self.hash)), "w"
            )
            csv_file.write("epoch, train-loss, val-loss, train-acc, val-acc\n")
            for i in range(len(hist["val-acc"])):
                csv_file.write(
                    "{},{},{},{},{}\n".format(
                        i,
                        hist["train-loss"][i],
                        hist["val-loss"][i],
                        hist["train-acc"][i],
                        hist["val-acc"][i],
                    )
                )
            csv_file.close()

    def __visualize_individual_img(self):
        """
        Private method that plots the network image output for a sample selected
        by the user in the console

        """
        if self.test_setup["visualize-individual-img"].lower() == "yes":

            names_save_files = []
            x, y, slice = None, None, None
            id_img = ""

            print("VISUALIZATION OF INDIVIDUAL IMAGES...")
            print("Available scans in the test set:")

            if self.train_setup["database"].lower() in ["mayo-challenge"]:
                for k in range(len(self.test_gen.scans)):
                    if k % 10 == 0:
                        print("\n")
                    print("{}       ".format(self.test_gen.scans[k]), end="")
                print("\n")

                scan = input("Which scan would you like to inspect? ")
                print(f"\nAvailable configs: {self.test_gen.configs}")
                config = input("Which config setup are you looking for? ")
                print("The following slices are available for scan {}:".format(scan))
                list_slices = self.test_gen.get_slices_available(scan, config)
                list_slices = list(map(lambda x: int(x.split("_")[1]), list_slices))
                for k, s_id in enumerate(sorted(list_slices)):
                    print(f"{s_id}\t", end="\n" if k % 7 == 0 else "")
                slice = input("\nWhich the slice number you would like to inspect? ")
                x, y = self.test_gen.get_specific_data(
                    scan,
                    int(slice),
                    config,
                    patch_size=int(self.test_setup["patch-size"]),
                )
                epo = (
                    int(self.test_management["load-specific-epoch"])
                    if self.test_management["load-specific-epoch"] != "none"
                    else -1
                )
                id_img = f"{scan}_{config.replace(' ', '')}_{slice}_epo{epo}"

                names_save_files.append(f"{str(self.hash)}_input_{scan}_{slice}.png")
                names_save_files.append(f"{str(self.hash)}_gt_{scan}_{slice}.png")
                names_save_files.append(f"{str(self.hash)}_pred_{scan}_{slice}.png")

            pred = self.model(x, training=False).numpy()

            # if sample values was normalized between -1 and 1
            if self.train_setup["negative_values"].lower() == "yes":
                x = (x + 1) / 2
                y = (y + 1) / 2
                pred = (pred + 1) / 2

            # To avoid overflow in casting from float32 to uint8
            x = np.minimum(np.maximum(x, 0.0), 1.0)
            y = np.minimum(np.maximum(y, 0.0), 1.0)
            pred = np.minimum(np.maximum(pred, 0.0), 1.0)

            x_show = (x[0, :, :, 0] * 255.0).astype(dtype=np.uint8)
            pred_show = (pred[0, :, :, 0] * 255.0).astype(dtype=np.uint8)
            y_show = (y[0, :, :, 0] * 255.0).astype(dtype=np.uint8)

            x_show = np.repeat(x_show[:, :, np.newaxis], 3, axis=2)
            pred_show = np.repeat(pred_show[:, :, np.newaxis], 3, axis=2)
            y_show = np.repeat(y_show[:, :, np.newaxis], 3, axis=2)

            textIn, textPred = "", ""
            textIn += f" PSNR:  {peak_signal_noise_ratio(y_show, x_show):.2f}"
            textPred += f" PSNR:  {peak_signal_noise_ratio(y_show, pred_show):.2f}"
            textIn += (
                f" SSIM:  {structural_similarity(y_show, x_show, channel_axis=-1):.4f}"
            )
            textPred += f" SSIM:  {structural_similarity(y_show, pred_show, channel_axis=-1):.4f}"
            textIn += f" NRMSE: {normalized_root_mse(y_show, x_show):.4f}"
            textPred += f" NRMSE: {normalized_root_mse(y_show, pred_show):.4f}"

            fig = make_subplots(
                rows=1, cols=3, subplot_titles=("Input", "Prediction", "Ground Truth")
            )
            fig.add_trace(go.Image(z=(np.repeat(x_show, 3, axis=-1))), row=1, col=1)
            fig.add_trace(go.Image(z=(np.repeat(pred_show, 3, axis=-1))), row=1, col=2)
            fig.add_trace(go.Image(z=(np.repeat(y_show, 3, axis=-1))), row=1, col=3)

            fig["layout"].update(
                annotations=[
                    fig["layout"]["annotations"][0],
                    fig["layout"]["annotations"][1],
                    fig["layout"]["annotations"][2],
                    dict(
                        x=int(self.test_setup["patch-size"]) // 2,
                        y=10,
                        showarrow=False,
                        text=textIn,
                        font_size=12,
                        font_color="yellow",
                        xref="x1",
                        yref="y1",
                    ),
                    dict(
                        x=int(self.test_setup["patch-size"]) // 2,
                        y=10,
                        showarrow=False,
                        text=textPred,
                        font_size=12,
                        font_color="yellow",
                        xref="x2",
                        yref="y2",
                    ),
                ]
            )

            dir_save_plot = os.path.join(
                self.management["output-dir-imgs"], f"{self.hash}_image_{id_img}.html"
            )
            print(f"SAVING PLOT OF SPECIFIC IMAGE IN: {dir_save_plot}")

            save_plot(fig, filename=dir_save_plot)
            fig.show()  # Maybe it can be removed

            save = input("Save images (input, pred and gt) separately? (yes/no) ")
            if save.lower() in ["yes", "y", "sim", "s"]:
                create_folder(self.management["output-dir-imgs"])

                imageio.imsave(
                    os.path.join(
                        self.management["output-dir-imgs"], names_save_files[0]
                    ),
                    x_show,
                )
                imageio.imsave(
                    os.path.join(
                        self.management["output-dir-imgs"], names_save_files[1]
                    ),
                    y_show,
                )
                imageio.imsave(
                    os.path.join(
                        self.management["output-dir-imgs"], names_save_files[2]
                    ),
                    pred_show,
                )

    def __compute_metrics(self):
        """
        Private method that writes the 'image_metrics_test_<hash>.csv' file with
        the SSIM, PSNR, and NRMSE computed for each image in the test file
        individually.
        """

        if self.test_setup["compute-metrics-all-set"].lower() not in [
            "no",
            "none",
            "n",
            "",
        ]:
            if self.train_setup["database"].lower() in ["mayo-challenge"]:
                data_gen = None
                if self.test_setup["compute-metrics-all-set"].lower() == "validation":
                    data_gen = self.val_gen
                elif self.test_setup["compute-metrics-all-set"].lower() == "test":
                    data_gen = self.test_gen
                else:
                    raise Exception("Partition for computing metrics is not defined")

                name_csv_file = "./outputs/image_metrics_{}_{}.csv".format(
                    self.test_setup["compute-metrics-all-set"].lower(), str(self.hash)
                )
                csv_file = open(name_csv_file, "w")
                csv_file.write(
                    "image,SSIM input,SSIM output,PSNR input,PSNR output,NRMSE input,NRMSE output,HFS input,HFS output\n"
                )
                for i, scan in enumerate(data_gen.scans):
                    print(
                        f"\nCOMPUTING METRICS: scan {i + 1}/{len(data_gen.scans)}",
                        end="",
                    )
                    for count_c, c in enumerate(data_gen.configs):
                        print(f"\nConfig: {count_c + 1}/{len(data_gen.configs)}")
                        pb = Progbar(len(data_gen.get_slices_available(scan, c)))
                        for j, slice_name in enumerate(
                            data_gen.get_slices_available(scan, c)
                        ):
                            x, y = data_gen.get_specific_data(
                                scan,
                                int(slice_name.split("_")[1]),
                                c,
                                patch_size=int(self.test_setup["patch-size"]),
                            )
                            pred = self.model(x, training=False).numpy()

                            # if sample values was normalized between -1 and 1
                            if self.train_setup["negative_values"].lower() == "yes":
                                x = (x + 1) / 2
                                y = (y + 1) / 2
                                pred = (pred + 1) / 2

                            # To avoid overflow in casting from float32 to uint8
                            x = np.minimum(np.maximum(x, 0.0), 1.0)
                            y = np.minimum(np.maximum(y, 0.0), 1.0)
                            pred = np.minimum(np.maximum(pred, 0.0), 1.0)

                            x = (x[0, :, :, 0] * 255.0).astype(dtype=np.uint8)
                            pred = (pred[0, :, :, 0] * 255.0).astype(dtype=np.uint8)
                            y = (y[0, :, :, 0] * 255.0).astype(dtype=np.uint8)

                            x = np.repeat(x[:, :, np.newaxis], 3, axis=2)
                            pred = np.repeat(pred[:, :, np.newaxis], 3, axis=2)
                            y = np.repeat(y[:, :, np.newaxis], 3, axis=2)

                            line = "{} {} {},".format(scan, c, slice_name)
                            line += "{:.4f},".format(
                                structural_similarity(y, x, channel_axis=-1)
                            )
                            line += "{:.4f},".format(
                                structural_similarity(y, pred, channel_axis=-1)
                            )
                            line += "{:.2f},".format(peak_signal_noise_ratio(y, x))
                            line += "{:.2f},".format(peak_signal_noise_ratio(y, pred))
                            line += "{:.4f},".format(normalized_root_mse(y, x))
                            line += "{:.4f},".format(normalized_root_mse(y, pred))
                            csv_file.write(line)
                            if j != 0 and j % 5 == 0:
                                pb.update(j)
                    print()
                csv_file.close()

                print(f"File: {name_csv_file}\nKey\t\t[MEAN   (STD)]")
                df = pd.read_csv(name_csv_file, sep=",")
                for k in df.keys()[1:]:
                    m = f"{np.mean(df[k].values):.4f}".replace(".", ",")
                    s = f"{np.std(df[k].values):.4f}".replace(".", ",")
                    print(f"{k}:\t{m} ({s})")

    def __test(self):
        """
        Private method executed when starting a new test mode. It draw plots,
        display images, and/or compute image metrics according to the user
        selection in the configuration file `.ini`.
        """
        with open("./outputs/logs/train-hist.{}".format(str(self.hash))) as json_file:
            train_history = json.load(json_file)
        self.test_setup = self.config["Test-Setup"]
        self.test_management = self.config["Experiment-Management"]
        if (
            self.test_management["load-specific-epoch"].lower() not in ["no", "none"]
            and self.train_setup["overwrite-weights-file"].lower() == "no"
        ):

            epoch = self.test_management["load-specific-epoch"]
            print(f"\nLoading a specific weights file. Epoch: {epoch}\n")
            if not epoch.isnumeric():
                raise f"Specify a number in 'load-specific-epoch' parameter in the test .ini file."
            epoch = int(epoch)
            self.model.load_weights(f"./weights/{self.hash}_{epoch}")
        else:
            print("\nLoading a unique/last weights file.\n")
            self.model.load_weights("./weights/{}".format(str(self.hash)))
        self.__hist_curves(train_history)
        self.__visualize_individual_img()
        self.__compute_metrics()


if __name__ == "__main__":
    start_time = time.time()
    exp = Experiment("train_experiment.ini")
    exp.load_data()
    exp.load_model()
    exp.execute()
    print(f"\n\nEXPERIMENT {exp.hash} FINISHED!")
    print("Hour now: ", str(datetime.datetime.now()), "\n")
    print(f"Total time elapsed: {time.time() - start_time:.2f} seconds")
