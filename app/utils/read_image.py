import skimage


def read_as_skimg(url: str):
    return skimage.io.imread(url)
