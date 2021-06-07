# TODO SUPPORT PNG
def getImageUrls(folder):
    # if path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
    return glob.glob(os.path.join(folder, '*.jpg'))


def getD2Urls(folder):
    return glob.glob(os.path.join(folder, '*.d2-net'))
