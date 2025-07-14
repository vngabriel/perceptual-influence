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

from src.data.mayo_challenge import MayoChallengeDataGenerator
from src.losses.perceptual_loss import PerceptualLoss
from src.models.unet import Unet
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
        self.affirmation_words = ["yes", "y", "sim", "s"]
        self.negation_words = ["no", "n", "none", ""]

        create_folder("./outputs")
        create_folder("./weights/")
        if (
            self.management["mode"].lower() == "test"
            or self.management["resume-previous-exp"].lower() in self.affirmation_words
        ):
            self.__resume_previous_experiment()
        else:
            self.__setup_new_experiment()

    def __resume_previous_experiment(self):
        self.hash = self.management["previous-hash"]
        print(f"LOADING WEIGHTS FROM THE PREVIOUS EXECUTION {self.hash}")
        with open(f"./outputs/train-setup.{self.hash}") as json_file:
            self.train_setup = json.load(json_file)

            if self.train_setup["seed"].lower() not in ["none", "", " "]:
                self.seed = int(self.train_setup["seed"])
                self.__set_experiment_seed()

        with open(f"./outputs/network-setup.{self.hash}") as json_file:
            self.network_setup = json.load(json_file)

        if os.path.exists(f"./outputs/loss-setup.{str(self.hash)}"):
            with open(f"./outputs/loss-setup.{self.hash}") as json_file:
                self.loss_setup = json.load(json_file)

    def __setup_new_experiment(self):
        if self.hash is None:
            self.hash = random.getrandbits(32)

        print(f"NEW EXPERIMENT HASH: {self.hash}")

        if self.management["mode"].lower() == "train":
            print("STARTING TRAINING MODE")
            self.train_setup = self.config["Train-Setup"]

            if self.train_setup["seed"].lower() not in ["none", "", " "]:
                self.seed = int(self.train_setup["seed"])
                self.__set_experiment_seed()

            self.network_setup = self.config["Network-Hyperparameters"]

            if self.train_setup["loss"].lower() in ["perceptual"]:
                self.loss_setup = self.config[
                    f"{self.train_setup['loss'].capitalize()}-Loss-Hyperparameters"
                ]

            with open(f"./outputs/train-setup.{self.hash}", "w") as outfile:
                json.dump(dict(self.train_setup), outfile, indent=2)

            with open(f"./outputs/network-setup.{self.hash}", "w") as outfile:
                json.dump(dict(self.network_setup), outfile, indent=2)

            if self.loss_setup is not None:
                with open(f"./outputs/loss-setup.{self.hash}", "w") as outfile:
                    json.dump(dict(self.loss_setup), outfile, indent=2)
        else:
            print("\nNOTHING TO DO. IS NOT POSSIBLE TEST A UNTRAINED MODEL!")
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
                f"./src/data/mayo-challenge-split.{self.train_setup['split-hash']}"
            ) as json_file:
                data = json.load(json_file)

            negative_normalize = (
                self.train_setup["negative_values"].lower() in self.affirmation_words
            )
            if self.management["mode"].lower() == "train":
                self.train_gen = MayoChallengeDataGenerator(
                    data_src=self.management["data-dir"],
                    scans=data["train"],
                    batch_size=int(self.train_setup["batch"]),
                    patch_size=int(self.train_setup["patch-size"]),
                    patch_stride=int(self.train_setup["patch-skip"]),
                    normalize_0_1=False,
                    negative_normalize=negative_normalize,
                    seed=self.seed,
                )
                self.val_gen = MayoChallengeDataGenerator(
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
                self.val_gen = MayoChallengeDataGenerator(
                    data_src=self.management["data-dir"],
                    scans=data["validation"],
                    batch_size=int(self.train_setup["batch"]),
                    patch_size=int(self.train_setup["patch-size"]),
                    patch_stride=int(self.train_setup["patch-skip"]),
                    normalize_0_1=False,
                    negative_normalize=negative_normalize,
                    seed=self.seed,
                )
                self.test_gen = MayoChallengeDataGenerator(
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

    def __load_loss(self):
        if self.train_setup["loss"].lower() == "mse":
            self.loss = tf.keras.losses.MeanSquaredError()
        elif self.train_setup["loss"].lower() == "mae":
            self.loss = tf.keras.losses.MeanAbsoluteError()
        elif self.train_setup["loss"].lower() == "perceptual":
            print(
                f"Using Perceptual Weight: {float(self.loss_setup['perceptual-weight'])}"
            )
            print(f"Using Content Layer: {self.loss_setup['perceptual-content-layer']}")
            print(f"Perceptual model: {self.loss_setup['perceptual-model']}")
            print(f"Weights path: {self.loss_setup['weights-path']}")
            self.loss = PerceptualLoss(
                input_size=self.train_setup["patch-size"],
                perceptual_space_loss=self.loss_setup["perceptual-space-loss"],
                image_space_loss=self.loss_setup["image-space-loss"],
                perceptual_weight=float(self.loss_setup["perceptual-weight"]),
                content_layer=self.loss_setup["perceptual-content-layer"],
                perceptual_model=self.loss_setup["perceptual-model"],
                weights_path=self.loss_setup["weights-path"],
            )
        else:
            raise Exception("Loss Function not available")

    def load_model(self):
        negative_output = (
            self.train_setup["negative_values"].lower() in self.affirmation_words
        )
        if self.train_setup["network"].lower() == "unet":
            self.model = Unet(
                output_activation="tanh" if negative_output else "sigmoid",
                channel=int(self.network_setup["channels"]),
                seed=self.seed,
            )
        else:
            raise Exception(f"Network not available: {self.train_setup['network']}")

        _ = self.model(
            np.zeros(self.val_gen.__getitem__(0)[0].shape), training=True
        )  # build
        self.model.summary()

        self.__load_loss()

    @tf.function
    def __train_step(self, x, y):
        with tf.GradientTape() as tape:
            pred = self.model(x, training=True)
            err = self.loss(y, pred)

        if self.train_setup["loss"].lower() == "perceptual":
            components = self.loss.get_loss_components()
        else:
            components = {}

        grads = tape.gradient(err, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.train_acc.update_state(y, pred)

        return err, components

    @tf.function
    def __val_step(self, x, y):
        pred = self.model(x, training=False)
        err = self.loss(y, pred)

        if self.train_setup["loss"].lower() == "perceptual":
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
        if self.management["mode"].lower() != "train":
            raise Exception("Impossible strating training outside the train mode")

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=float(self.train_setup["lr"])
        )
        self.train_acc = tf.keras.metrics.CosineSimilarity(name="train_acc")
        self.val_acc = tf.keras.metrics.CosineSimilarity(name="val_acc")

        first_epoch = 0
        training_time = 0.0

        if self.management["resume-previous-exp"].lower() in self.affirmation_words:
            first_epoch = int(self.management["last-epoch-executed"])
            with open(f"./outputs/logs/train-hist.{self.hash}") as json_file:
                train_history = json.load(json_file)

            if (
                self.train_setup["overwrite-weights-file"].lower()
                in self.affirmation_words
            ):
                self.model.load_weights(f"./weights/{self.hash}")
            else:
                self.model.load_weights(f"./weights/{self.hash}_{first_epoch}")

            print(f"RESTARTING {self.hash} FROM EPOCH {first_epoch + 1}")
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
            if self.train_setup["loss"].lower() == "perceptual":
                train_history["train-image-loss"] = []
                train_history["train-perceptual-loss"] = []
                train_history["val-image-loss"] = []
                train_history["val-perceptual-loss"] = []

        for epoch in range(first_epoch, int(self.train_setup["epochs"])):
            print(f"\nTRAINING: epoch {epoch + 1}/{self.train_setup['epochs']}")

            start = time.time()
            pb_train = Progbar(
                len(self.train_gen) * self.train_gen.batch_size,
                stateful_metrics=["train-loss", "train-acc"],
            )

            if self.train_setup["loss"].lower() == "perceptual":
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

                if self.train_setup["loss"].lower() == "perceptual":
                    total_train_image_loss += train_components["image_loss"]
                    total_train_perceptual_loss += train_components["perceptual"]

                if i >= 0:
                    if self.train_setup["loss"].lower() == "perceptual":
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

            print(f"VALIDATING: epoch {epoch + 1}/{self.train_setup['epochs']}")
            pb_val = Progbar(
                len(self.val_gen) * self.val_gen.batch_size,
                stateful_metrics=["val-loss", "val-acc"],
            )

            if self.train_setup["loss"].lower() == "perceptual":
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

                if self.train_setup["loss"].lower() == "perceptual":
                    total_val_image_loss += val_components["image_loss"]
                    total_val_perceptual_loss += val_components["perceptual"]

                if i >= 0:
                    if self.train_setup["loss"].lower() == "perceptual":
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
            print("")

            if (
                self.train_setup["overwrite-weights-file"].lower()
                in self.negation_words
            ):
                self.model.save_weights(f"./weights/{self.hash}_{epoch + 1}")
                print(f"Last epoch ({epoch}) saved (weight file was NO overwriten)")
            else:
                self.model.save_weights(f"./weights/{self.hash}")
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

            if self.train_setup["loss"].lower() == "perceptual":
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
            with open(f"./outputs/logs/train-hist.{self.hash}", "w") as outfile:
                json.dump(train_history, outfile, indent=2)

            self.train_acc.reset_states()
            self.val_acc.reset_states()

    def __hist_curves(self, hist):
        print(
            "PLOTTING THE ACCURACY AND/OR THE LOSS CURVES DURING THE PREVIOUS TRAINING..."
        )

        create_folder(self.management["output-dir-imgs"])

        data = pd.DataFrame(hist)
        show_plot = False
        plot_data = []

        if self.test_setup["train-loss-curve"].lower() in self.affirmation_words:
            plot_data.append("train-loss")
            show_plot = True
        if self.test_setup["val-loss-curve"].lower() in self.affirmation_words:
            plot_data.append("val-loss")
            show_plot = True
        if self.test_setup["train-acc-curve"].lower() in self.affirmation_words:
            plot_data.append("train-acc")
            show_plot = True
        if self.test_setup["val-acc-curve"].lower() in self.affirmation_words:
            plot_data.append("val-acc")
            show_plot = True

        if show_plot:
            fig = px.line(data, y=plot_data, title="Train History")
            dir_save_plot = os.path.join(
                self.management["output-dir-imgs"], f"{self.hash}_hist.html"
            )
            print(f"SAVING PLOT OF TRAIN HISTORY IN: {dir_save_plot}")

            save_plot(fig, filename=dir_save_plot)
            fig.show()

        if self.test_setup["save-curves-as-csv"].lower() in self.affirmation_words:
            csv_file = open(f"./outputs/train_hist_curves-{self.hash}.csv", "w")
            csv_file.write("epoch, train-loss, val-loss, train-acc, val-acc\n")
            for i in range(len(hist["val-acc"])):
                csv_file.write(
                    f"{i},{hist['train-loss'][i]},{hist['val-loss'][i]},{hist['train-acc'][i]},{hist['val-acc'][i]}\n"
                )
            csv_file.close()

    def __post_process_images_for_evaluation(self, x, y, pred):
        if self.train_setup["negative_values"].lower() in self.affirmation_words:
            x = (x + 1) / 2
            y = (y + 1) / 2
            pred = (pred + 1) / 2

        x = np.minimum(np.maximum(x, 0.0), 1.0)
        y = np.minimum(np.maximum(y, 0.0), 1.0)
        pred = np.minimum(np.maximum(pred, 0.0), 1.0)

        x = (x[0, :, :, 0] * 255.0).astype(dtype=np.uint8)
        y = (y[0, :, :, 0] * 255.0).astype(dtype=np.uint8)
        pred = (pred[0, :, :, 0] * 255.0).astype(dtype=np.uint8)

        x = np.repeat(x[:, :, np.newaxis], 3, axis=2)
        y = np.repeat(y[:, :, np.newaxis], 3, axis=2)
        pred = np.repeat(pred[:, :, np.newaxis], 3, axis=2)

        return x, y, pred

    def __prepare_images_for_visualization(self, x, y, pred):
        text_in, text_pred = "", ""
        text_in += f" PSNR:  {peak_signal_noise_ratio(y, x):.2f}"
        text_pred += f" PSNR:  {peak_signal_noise_ratio(y, pred):.2f}"
        text_in += f" SSIM:  {structural_similarity(y, x, channel_axis=-1):.4f}"
        text_pred += f" SSIM:  {structural_similarity(y, pred, channel_axis=-1):.4f}"
        text_in += f" NRMSE: {normalized_root_mse(y, x):.4f}"
        text_pred += f" NRMSE: {normalized_root_mse(y, pred):.4f}"

        fig = make_subplots(
            rows=1, cols=3, subplot_titles=("Input", "Prediction", "Ground Truth")
        )
        fig.add_trace(go.Image(z=(np.repeat(x, 3, axis=-1))), row=1, col=1)
        fig.add_trace(go.Image(z=(np.repeat(pred, 3, axis=-1))), row=1, col=2)
        fig.add_trace(go.Image(z=(np.repeat(y, 3, axis=-1))), row=1, col=3)

        fig["layout"].update(
            annotations=[
                fig["layout"]["annotations"][0],
                fig["layout"]["annotations"][1],
                fig["layout"]["annotations"][2],
                dict(
                    x=int(self.test_setup["patch-size"]) // 2,
                    y=10,
                    showarrow=False,
                    text=text_in,
                    font_size=12,
                    font_color="yellow",
                    xref="x1",
                    yref="y1",
                ),
                dict(
                    x=int(self.test_setup["patch-size"]) // 2,
                    y=10,
                    showarrow=False,
                    text=text_pred,
                    font_size=12,
                    font_color="yellow",
                    xref="x2",
                    yref="y2",
                ),
            ]
        )

        return fig

    def __visualize_individual_img(self):
        if (
            self.test_setup["visualize-individual-img"].lower()
            in self.affirmation_words
        ):
            names_save_files = []
            x, y, slice_ = None, None, None
            id_img = ""

            print("VISUALIZATION OF INDIVIDUAL IMAGES...")
            print("Available scans in the test set:")

            if self.train_setup["database"].lower() in ["mayo-challenge"]:
                for k in range(len(self.test_gen.scans)):
                    if k % 10 == 0:
                        print("\n")
                    print(f"{self.test_gen.scans[k]}       ", end="")
                print("\n")

                scan = input("Which scan would you like to inspect? ")
                print(f"\nAvailable configs: {self.test_gen.configs}")
                config = input("Which config setup are you looking for? ")
                print(f"The following slices are available for scan {scan}:")
                list_slices = self.test_gen.get_slices_available(scan, config)
                list_slices = list(map(lambda x: int(x.split("_")[1]), list_slices))

                for k, s_id in enumerate(sorted(list_slices)):
                    print(f"{s_id}\t", end="\n" if k % 7 == 0 else "")

                slice_ = input("\nWhich the slice number you would like to inspect? ")
                x, y = self.test_gen.get_specific_data(
                    scan,
                    int(slice_),
                    config,
                    patch_size=int(self.test_setup["patch-size"]),
                )
                epo = (
                    int(self.test_management["load-specific-epoch"])
                    if self.test_management["load-specific-epoch"] != "none"
                    else -1
                )
                id_img = f"{scan}_{config.replace(' ', '')}_{slice_}_epo{epo}"

                names_save_files.append(f"{str(self.hash)}_input_{scan}_{slice_}.png")
                names_save_files.append(f"{str(self.hash)}_gt_{scan}_{slice_}.png")
                names_save_files.append(f"{str(self.hash)}_pred_{scan}_{slice_}.png")

            pred = self.model(x, training=False).numpy()
            x_show, y_show, pred_show = self.__post_process_images_for_evaluation(
                x, y, pred
            )
            fig = self.__prepare_images_for_visualization(x_show, y_show, pred_show)

            dir_save_plot = os.path.join(
                self.management["output-dir-imgs"], f"{self.hash}_image_{id_img}.html"
            )
            print(f"SAVING PLOT OF SPECIFIC IMAGE IN: {dir_save_plot}")
            save_plot(fig, filename=dir_save_plot)
            fig.show()

            save = input("Save images (input, pred and gt) separately? (yes/no) ")
            if save.lower() in self.affirmation_words:
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

    def __get_evaluation_dataset(self):
        if self.train_setup["database"].lower() == "mayo-challenge":
            if self.test_setup["compute-metrics-all-set"].lower() == "validation":
                return self.val_gen
            elif self.test_setup["compute-metrics-all-set"].lower() == "test":
                return self.test_gen
            else:
                raise Exception("Partition for computing metrics is not defined")
        else:
            raise Exception("Database not available for evaluation")

    def __compute_metrics(self):
        if (
            self.test_setup["compute-metrics-all-set"].lower()
            not in self.negation_words
        ):
            data_gen = self.__get_evaluation_dataset()
            name_csv_file = f"./outputs/image_metrics_{self.test_setup['compute-metrics-all-set'].lower()}_{self.hash}.csv"
            csv_file = open(name_csv_file, "w")
            csv_file.write(
                "image,SSIM input,SSIM output,PSNR input,PSNR output,NRMSE input,NRMSE output\n"
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
                        x, y, pred = self.__post_process_images_for_evaluation(
                            x, y, pred
                        )

                        line = f"{scan} {c} {slice_name},"
                        line += f"{structural_similarity(y, x, channel_axis=-1):.4f},"
                        line += (
                            f"{structural_similarity(y, pred, channel_axis=-1):.4f},"
                        )
                        line += f"{peak_signal_noise_ratio(y, x):.2f},"
                        line += f"{peak_signal_noise_ratio(y, pred):.2f},"
                        line += f"{normalized_root_mse(y, x):.4f},"
                        line += f"{normalized_root_mse(y, pred):.4f}\n"
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
        with open(f"./outputs/logs/train-hist.{self.hash}") as json_file:
            train_history = json.load(json_file)

        self.test_setup = self.config["Test-Setup"]
        self.test_management = self.config["Experiment-Management"]
        if (
            self.test_management["load-specific-epoch"].lower()
            not in self.negation_words
            and self.train_setup["overwrite-weights-file"].lower()
            in self.negation_words
        ):
            epoch = self.test_management["load-specific-epoch"]
            print(f"\nLoading a specific weights file. Epoch: {epoch}\n")

            if not epoch.isnumeric():
                raise Exception(
                    "Specify a number in 'load-specific-epoch' parameter in the test .ini file."
                )

            epoch = int(epoch)
            self.model.load_weights(f"./weights/{self.hash}_{epoch}")
        else:
            print("\nLoading a unique/last weights file.\n")
            self.model.load_weights(f"./weights/{self.hash}")

        self.__hist_curves(train_history)
        self.__visualize_individual_img()
        self.__compute_metrics()


if __name__ == "__main__":
    start_time = time.time()

    experiment = Experiment("test_experiment.ini")
    experiment.load_data()
    experiment.load_model()
    experiment.execute()

    print(f"\n\nEXPERIMENT {experiment.hash} FINISHED!")
    print("Hour now: ", str(datetime.datetime.now()), "\n")
    print(f"Total time elapsed: {time.time() - start_time:.2f} seconds")
