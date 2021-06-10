import numpy as np
import math
import cv2
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
import cv2

def padding(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride) # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad

# transfer caffe model to pytorch which will match the layer name
def transfer(model, model_weights):
    transfered_model_weights = {}
    for weights_name in model.state_dict().keys():
        transfered_model_weights[weights_name] = model_weights['.'.join(weights_name.split('.')[1:])]
    return transfered_model_weights

# get max index of 2d array
def npmax(array):
    arrayindex = array.argmax(1)
    arrayvalue = array.max(1)
    i = arrayvalue.argmax()
    j = arrayindex[i]
    return i, j

# pose vis
def pose_vis(img, candidate, subset, al_list):
    # point color
    pose_color = np.random.rand(len(subset), 3) * 255

    # plot point
    for i in range(len(subset)):
        mid_x1 = candidate[subset[i][8].astype(int)][0]
        mid_x2 = candidate[subset[i][11].astype(int)][0]
        mid_y1 = candidate[subset[i][8].astype(int)][1]
        mid_y2 = candidate[subset[i][11].astype(int)][1]
        mid_x = (mid_x1 + mid_x2) / 2
        mid_y = (mid_y1 + mid_y2) / 2
        img = cv2.putText(img, str(i+1), (mid_x.astype(int), mid_y.astype(int)),
                          cv2.FONT_HERSHEY_COMPLEX,2,pose_color[i],5)
        for jj in range(18):
            if subset[i][jj] > -1:
                img = cv2.circle(img, (candidate[subset[i][jj].astype(int)][0].astype(int),
                                       candidate[subset[i][jj].astype(int)][1].astype(int)),
                                 radius=4, color=pose_color[i], thickness=2)


    return img

