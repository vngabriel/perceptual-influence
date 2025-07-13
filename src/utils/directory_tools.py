import os
import glob


def clear_folder(dir_f, recursive=False, folders=False):
    if len(dir_f) == 1 and dir_f[0] == "/":
        print("May I remore '/'?")
        exit()

    if recursive:
        files = glob.glob(os.path.join(dir_f, "**/*.*"), recursive=True)
        for f in files:
            os.remove(f)  # files

        if folders:
            files = glob.glob(os.path.join(dir_f, "**/*"), recursive=True)
            files = sorted(files, key=lambda x: len(x), reverse=True)
            for f in files:
                os.rmdir(f)  # folders
    else:
        files = glob.glob(os.path.join(dir_f, "*.*"), recursive=True)
        for f in files:
            os.remove(f)  # just files


def create_folder(dir_f):
    if not os.path.exists(dir_f):
        os.makedirs(dir_f)
