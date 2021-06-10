import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
from utils import model
from utils import util
from utils.pose import pose_detector
from utils import model
import os

def main_exe():
    # load image
    img_path  = input("The full image path: ")
    # img_path = os.path.join(".","images","lift.jpg")

    if os.path.exists(img_path):
        print("image loaded")
    else:
        print("image not exists")
        return
    img = cv2.imread(img_path)  # B,G,R order

    # check gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)

    # load rula estimator
    model_rula = model.NN_base().to(device)
    model_rula.load_state_dict(torch.load(os.path.join('.', 'weights', "rula.pth")))

    # load pose detector
    pose_estimation = pose_detector(os.path.join(".","weights","pose_coco.pth"))

    ## pose detection
    candidate, subset = pose_estimation(img)
    # print(subset)


    if len(subset) < 1:
        print("not detected")
        return
    else:
        al_list = []
        for i in range(len(subset)):
            if subset[i][8] == -1 or subset[i][11] == -1:
                print("The "+str(i+1)+" th person has missing points")
                break
            mid_x1 = candidate[subset[i][8].astype(int)][0]
            mid_x2 = candidate[subset[i][11].astype(int)][0]
            mid_y1 = candidate[subset[i][8].astype(int)][1]
            mid_y2 = candidate[subset[i][11].astype(int)][1]
            mid_x = (mid_x1 + mid_x2) / 2
            mid_y = (mid_y1 + mid_y2) / 2

            pose_xy = -1*np.ones((18,2))
            for jj in range(18):
                if subset[i][jj] > -1:
                    pose_xy[jj][:] = candidate[subset[i][jj].astype(int)][:2]
                    pose_xy[jj][0] -= mid_x
                    pose_xy[jj][1] -= mid_y
            pose_xy = pose_xy.reshape((1, 36))
            y_pred = model_rula(torch.tensor(pose_xy).float().to(device))
            # print(y_pred)

            # predicted action level
            y_pred = torch.argmax(y_pred)
            al_list.append(y_pred.item())
            print("Action level of the "+str(i+1)+" th person: " +str(y_pred.item()+1))

        # plot and show
        img = util.pose_vis(img, candidate, subset,al_list)
        cv2.imshow("test", img)
        cv2.waitKey(0)

if __name__ == '__main__':
    main_exe()