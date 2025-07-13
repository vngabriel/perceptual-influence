"""
URL of this script:
https://github.com/SSinyu/RED-CNN/blob/master/prep.py

Credits:
https://github.com/SSinyu

See more:
https://github.com/SSinyu/RED-CNN
"""

import os
import argparse
import numpy as np
import pydicom


def save_dataset(args):
    args.save_path = os.path.join(args.save_path, f"{args.mm}mm {args.kernel}")
    args.data_path = os.path.join(args.data_path, f"{args.mm}mm {args.kernel}")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print(f"Create path : {args.save_path}")

    patients_list = sorted([d for d in os.listdir(args.data_path) if "zip" not in d])
    for p_ind, patient in enumerate(patients_list):

        if args.kernel == "D45":
            patient_input_path = os.path.join(
                args.data_path, patient, f"quarter_{args.mm}mm_sharp"
            )
            patient_target_path = os.path.join(
                args.data_path, patient, f"full_{args.mm}mm_sharp"
            )
        else:
            patient_input_path = os.path.join(
                args.data_path, patient, f"quarter_{args.mm}mm"
            )
            patient_target_path = os.path.join(
                args.data_path, patient, f"full_{args.mm}mm"
            )

        for path_ in [patient_input_path, patient_target_path]:
            full_pixels = get_pixels_hu(load_scan(path_))
            for pi in range(len(full_pixels)):
                io = "input" if "quarter" in path_ else "target"
                f = normalize_(
                    full_pixels[pi], args.norm_range_min, args.norm_range_max
                )
                f_name = f"{patient}_{pi}_{io}.npy"
                np.save(os.path.join(args.save_path, f_name), f)

        show_progress(
            p_ind,
            len(patients_list),
            start_zero=True,
            reset_line=True,
            suffix=f"Processing {patient}",
        )
    print()


def load_scan(path):
    # referred from https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
    slices = [pydicom.read_file(os.path.join(path, s)) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    try:
        slice_thickness = np.abs(
            slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2]
        )
    except Exception:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def get_pixels_hu(slices):
    # referred from https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    image[image == -2000] = 0

    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def normalize_(image, min_b=-1024.0, max_b=3072.0):
    image = (image - min_b) / (max_b - min_b)

    return image


def show_progress(
    ite, num_iters, start_zero=True, suffix="", bar_size=30, reset_line=False
):
    if start_zero and num_iters > 1:
        num_iters -= 1
    else:
        start_zero = False

    okay = ite / num_iters
    print(
        f"Progress |{'=' * round(okay * bar_size)}{'-' * round((1 - okay) * bar_size)}|",
        end=" ",
    )
    print(f"{okay*100 :.1f}% complete", end=" ")

    if start_zero:
        print(f"[{ite + 1}/{num_iters + 1}]", end=" ")
    else:
        print(f"[{ite}/{num_iters}]", end=" ")

    print(f"{suffix}", end="\r" if reset_line else "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    src_folder = "/home/gabriel/Research/dataset/mayo-challenge-raw/"

    parser.add_argument("--mm", type=int, default=1)
    parser.add_argument("--kernel", type=str, default="B30")
    parser.add_argument(
        "--data_path", type=str, default=f"{src_folder}Training_Image_Data/"
    )
    parser.add_argument("--save_path", type=str, default=f"{src_folder}/npy_img/")

    parser.add_argument("--test_patient", type=str, default="L506")
    parser.add_argument("--norm_range_min", type=float, default=-1024.0)
    parser.add_argument("--norm_range_max", type=float, default=3072.0)

    args = parser.parse_args()
    save_dataset(args)
