import os
import wget
import zipfile


def dl_trained_params(url="https://tinyurl.com/3pvy4d4k"):
    filename = wget.download(url)
    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall("trained_params")
    if os.path.isfile(filename):
        os.remove(filename)

