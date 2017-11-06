from scipy import misc
import numpy as np


def load_image(im_path, read_channel=None, resize=None):
    # im = cv2.imread(im_path, self._cv_read)
    if read_channel is None:
        im = misc.imread(im_path)
    elif read_channel == 3:
        im = misc.imread(im_path, mode='RGB')
    else:
        im = misc.imread(im_path, flatten=True)

    if len(im.shape) < 3:
        try:
            im = misc.imresize(im, (resize[0], resize[1], 1))
        except TypeError:
            pass
        im = np.reshape(im, [1, im.shape[0], im.shape[1], 1])
    else:
        try:
            im = misc.imresize(im, (resize[0], resize[1], im.shape[2]))
        except TypeError:
            pass
        im = np.reshape(im, [1, im.shape[0], im.shape[1], im.shape[2]])
    return im
