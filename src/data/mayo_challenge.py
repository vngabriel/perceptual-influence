import tensorflow as tf
import numpy as np
import os
import random


class MayoChallengeDataGenerator(tf.keras.utils.Sequence):
    """Mayo Challenge sample values are already between 0 and 1. But majority hasn't 0 ou 1 value."""

    def __init__(
        self,
        data_src,
        scans,
        batch_size,
        patch_size,
        patch_stride,
        normalize_0_1,
        negative_normalize,
        seed,
    ):
        self.image_size = 512
        self.batch_size = batch_size
        self.total_samples = []
        self.scans = scans
        self.configs = ["3mm D45", "3mm B30", "1mm D45", "1mm B30"]

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

    def get_specific_data(self, scan, slice_, config, patch_size=None):
        idx = [
            index
            for index, (folder, file) in enumerate(self.total_samples)
            if (file.startswith(scan) and int(file.split("_")[1]) == slice_)
            and folder.endswith(config)
        ]

        if patch_size is None:
            x = np.zeros((1, self.image_size, self.image_size, 1), dtype=np.float16)
            y = np.zeros((1, self.image_size, self.image_size, 1), dtype=np.float16)
        else:
            x = np.zeros((1, patch_size, patch_size, 1), dtype=np.float16)
            y = np.zeros((1, patch_size, patch_size, 1), dtype=np.float16)

        tmp_x = np.load(
            os.path.join(self.total_samples[idx[0]][0], self.total_samples[idx[0]][1])
        )
        tmp_y = np.load(
            os.path.join(
                self.total_samples[idx[0]][0],
                self.total_samples[idx[0]][1].replace("input", "target"),
            )
        )

        if patch_size is not None:
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

        if self.normalize_0_1:
            x[0, :, :, 0] = tmp_x / np.max(tmp_x) if np.max(tmp_x) != 0 else tmp_x
            y[0, :, :, 0] = tmp_y / np.max(tmp_y) if np.max(tmp_y) != 0 else tmp_y
        else:
            x[0, :, :, 0] = tmp_x
            y[0, :, :, 0] = tmp_y

        if self.negative_normalize:
            x = (x * 2) - 1
            y = (y * 2) - 1

        return x, y

    def __get_data(self, batch):
        x = np.zeros(
            (len(batch), self.patch_size, self.patch_size, 1), dtype=np.float16
        )
        y = np.zeros(
            (len(batch), self.patch_size, self.patch_size, 1), dtype=np.float16
        )

        for i, b in enumerate(batch):
            tmp_x = np.load(os.path.join(b[0][0], b[0][1]))
            tmp_y = np.load(os.path.join(b[0][0], b[0][1].replace("input", "target")))

            if self.normalize_0_1:
                tmp_x = tmp_x / np.max(tmp_x) if np.max(tmp_x) > 0 else tmp_x
                tmp_y = tmp_y / np.max(tmp_y) if np.max(tmp_y) > 0 else tmp_y

            if self.negative_normalize:
                tmp_x = (tmp_x * 2) - 1
                tmp_y = (tmp_y * 2) - 1

            x[i, :, :, 0] = tmp_x[
                b[1][0] : b[1][0] + self.patch_size, b[1][1] : b[1][1] + self.patch_size
            ]
            y[i, :, :, 0] = tmp_y[
                b[1][0] : b[1][0] + self.patch_size, b[1][1] : b[1][1] + self.patch_size
            ]

        return x, y
