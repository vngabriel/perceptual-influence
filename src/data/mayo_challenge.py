import tensorflow as tf
import numpy as np
import os
import random


class MayoChallengeDataGenerator(tf.keras.utils.Sequence):
    """
    Mayo Challenge sample values is already between 0 and 1. But majority hasn't
    0 ou 1 values.
    """

    def __init__(
        self,
        data_src,
        scans,
        batch_size=10,
        patch_size=55,
        patch_stride=10,
        normalize_0_1=False,
        negative_normalize=False,
        seed=None,
    ):

        self.image_size = 512
        self.batch_size = batch_size
        self.total_samples = []
        self.scans = scans
        self.configs = ["3mm D45", "3mm B30", "1mm D45", "1mm B30"]
        # self.configs = ['1mm B30']

        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        for folder in self.scans:
            for config in self.configs:
                files = os.listdir(os.path.join(data_src, "npy_img", config))
                for f in [
                    x for x in files if x.startswith(folder) and x.endswith("input.npy")
                ]:
                    self.total_samples.append(
                        (os.path.join(data_src, "npy_img", config), f)
                    )

        random.shuffle(self.total_samples)

        self.patch_size = patch_size
        self.all_patches_location = self.get_patches_ref(self.patch_size, patch_stride)

        self.normalize_0_1 = normalize_0_1
        self.negative_normalize = negative_normalize

    def get_patches_ref(self, size, stride):
        inits_left_upper_corner = [
            (u, v)
            for u in np.arange(0, self.image_size - size, stride)
            for v in np.arange(0, self.image_size - size, stride)
        ]
        ret = [(a, b) for a in self.total_samples for b in inits_left_upper_corner]
        return ret

    def __len__(self):
        return len(self.all_patches_location) // self.batch_size

    def __getitem__(self, index):
        batch = self.all_patches_location[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        return self.__get_data(batch)

    def get_slices_available(self, target_scan, config):
        return sorted(
            [
                file
                for (folder, file) in self.total_samples
                if (target_scan in file and config in folder)
            ]
        )

    def get_specific_data(self, scan, slice, config, patch_size=None):
        idx = [
            index
            for index, (folder, file) in enumerate(self.total_samples)
            if (file.startswith(scan) and int(file.split("_")[1]) == slice)
            and folder.endswith(config)
        ]

        if patch_size == None:
            X = np.zeros((1, self.image_size, self.image_size, 1), dtype=np.float16)
            Y = np.zeros((1, self.image_size, self.image_size, 1), dtype=np.float16)
        else:
            X = np.zeros((1, patch_size, patch_size, 1), dtype=np.float16)
            Y = np.zeros((1, patch_size, patch_size, 1), dtype=np.float16)

        tmp_x = np.load(
            os.path.join(self.total_samples[idx[0]][0], self.total_samples[idx[0]][1])
        )
        tmp_y = np.load(
            os.path.join(
                self.total_samples[idx[0]][0],
                self.total_samples[idx[0]][1].replace("input", "target"),
            )
        )

        if patch_size != None:
            tmp_x = tmp_x[
                self.image_size // 2
                - patch_size // 2 : self.image_size // 2
                + patch_size // 2,
                self.image_size // 2
                - patch_size // 2 : self.image_size // 2
                + patch_size // 2,
            ]
            tmp_y = tmp_y[
                self.image_size // 2
                - patch_size // 2 : self.image_size // 2
                + patch_size // 2,
                self.image_size // 2
                - patch_size // 2 : self.image_size // 2
                + patch_size // 2,
            ]

        # Normalize between 0 and 1
        # But: min(X) != 0; max(X) = 1
        if self.normalize_0_1:
            X[0, :, :, 0] = tmp_x / np.max(tmp_x) if np.max(tmp_x) != 0 else tmp_x
            Y[0, :, :, 0] = tmp_y / np.max(tmp_y) if np.max(tmp_y) != 0 else tmp_y
        else:
            # Don't normalize (output will be values between 0 and 1)
            # But: min(X) != 0; max(X) != 1
            X[0, :, :, 0] = tmp_x
            Y[0, :, :, 0] = tmp_y

        # Normalize beetween -1 and 1. Just: (w * 2) - 1
        if self.negative_normalize:
            X = (X * 2) - 1
            Y = (Y * 2) - 1

        return X, Y

    def __get_data(self, batch):
        X = np.zeros(
            (len(batch), self.patch_size, self.patch_size, 1), dtype=np.float16
        )
        Y = np.zeros(
            (len(batch), self.patch_size, self.patch_size, 1), dtype=np.float16
        )

        for i, b in enumerate(batch):

            tmp_x = np.load(os.path.join(b[0][0], b[0][1]))
            tmp_y = np.load(os.path.join(b[0][0], b[0][1].replace("input", "target")))

            # Normalize between 0 and 1
            # But: min(X) != 0; max(X) = 1
            if self.normalize_0_1:
                tmp_x = tmp_x / np.max(tmp_x) if np.max(tmp_x) > 0 else tmp_x
                tmp_y = tmp_y / np.max(tmp_y) if np.max(tmp_y) > 0 else tmp_y
            # else:     Don't normalize (output will be values between 0 and 1)
            #           But: min(X) != 0; max(X) != 1

            # Normalize beetween -1 and 1. Just: (w * 2) - 1
            if self.negative_normalize:
                tmp_x = (tmp_x * 2) - 1
                tmp_y = (tmp_y * 2) - 1

            X[i, :, :, 0] = tmp_x[
                b[1][0] : b[1][0] + self.patch_size, b[1][1] : b[1][1] + self.patch_size
            ]
            Y[i, :, :, 0] = tmp_y[
                b[1][0] : b[1][0] + self.patch_size, b[1][1] : b[1][1] + self.patch_size
            ]

        return X, Y

    # def get_numpy_example(self, network, batch_nr=2, img_nr=0):
    #     exX, exY = self.__getitem__(batch_nr)
    #     exX = tf.convert_to_tensor(exX, dtype=tf.float16)
    #     exPred = network(exX, training=False)

    #     exX = exX.numpy()
    #     exPred = exPred.numpy()

    # Attention with this normalization!!
    #     exX =    (exX - np.min(exX))       / (np.max(exX)    - np.min(exX))
    #     exPred = (exPred - np.min(exPred)) / (np.max(exPred) - np.min(exPred))
    #     exY =    (exY - np.min(exY))       / (np.max(exY)    - np.min(exY))

    #     return np.squeeze(exX[img_nr,:,:,0]), np.squeeze(exPred[img_nr,:,:,0]), np.squeeze(exY[img_nr,:,:,0])


if __name__ == "__main__":
    src = "/mnt/Baltz_Datasets/Mayo_Challenge"
    scans = ["L192"]

    data = MayoChallenge_DataGenerator(
        src, scans, normalize_0_1=False, negative_normalize=False
    )
    # print(data.total_samples)
    # print(data.get_slices_available('L067'))
    # print(data.all_patches_location)
    # print(len(data.all_patches_location))

    x, y = data.get_specific_data("L192", 1, "1mm B30", patch_size=256)

    print(y.shape)
    print("x min {} max {}".format(np.min(x), np.max(x)))
    print("y min {} max {}".format(np.min(y), np.max(y)))

    from matplotlib import pyplot as plt

    # fig = plt.figure()
    plt.figure("LOW DOSE")
    plt.imshow(x[0, :, :, 0], cmap="gray")
    plt.figure("HIGH DOSE")
    plt.imshow(y[0, :, :, 0], cmap="gray")
    plt.show()

    # import matplotlib
    # matplotlib.image.imsave("fota.png", np.repeat(x[0,:,:,:], 3, axis=2))
