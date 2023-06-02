from torch.nn.modules.module import Module
import torch.nn.functional as F
import numpy as np

class my_MaxPool2d(Module):

    def __init__(self, kernel_size, stride):
        super(my_MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input):

        input = input.transpose(3,1)
        input = F.max_pool2d(input, self.kernel_size, self.stride)
        input = input.transpose(3,1).contiguous()

        return input


def intersection_over_union(box1, box2, wh=False):
    """
    计算IoU（交并比）
    :param box1: bounding box1
    :param box2: bounding box2
    :param wh: 坐标的格式是否为（x,y,w,h）
    :return:计算结果
    """
    if not wh:
        xmin1, ymin1, xmax1, ymax1 = box1[:4]
        xmin2, ymin2, xmax2, ymax2 = box2[:4]
    else:
        xmin1, ymin1 = int(box1[0] - box1[2] / 2.0), int(box1[1] - box1[3] / 2.0)
        xmax1, ymax1 = int(box1[0] + box1[2] / 2.0), int(box1[1] + box1[3] / 2.0)
        xmin2, ymin2 = int(box2[0] - box2[2] / 2.0), int(box2[1] - box2[3] / 2.0)
        xmax2, ymax2 = int(box2[0] + box2[2] / 2.0), int(box2[1] + box2[3] / 2.0)
    # 获取矩形框交集对应的左上角和右下角的坐标（intersection）
    xx1 = max([xmin1, xmin2])
    yy1 = max([ymin1, ymin2])
    xx2 = min([xmax1, xmax2])
    yy2 = min([ymax1, ymax2])
    # 计算两个矩形框面积
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    inter_area = (max([0, xx2 - xx1])) * (max([0, yy2 - yy1]))  # 计算交集面积
    iou = inter_area / (area1 + area2 - inter_area)  # 计算交并比
    return iou

def GIoU(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2

    # IOU
    xx1 = np.maximum(b1_x1, b2_x1)
    yy1 = np.maximum(b1_y1, b2_y1)
    xx2 = np.minimum(b1_x2, b2_x2)
    yy2 = np.minimum(b1_y2, b2_y2)
    inter_w = np.maximum(0.0, yy2 - yy1)
    inter_h = np.maximum(0.0, xx2 - xx1)
    inter = inter_w * inter_h
    Union = (b1_x2 - b1_x1) * (b1_y2 - b1_y1) + (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter

    # GIOU
    C_xx1 = np.minimum(b1_x1, b2_x1)
    C_yy1 = np.minimum(b1_y1, b2_y1)
    C_xx2 = np.maximum(b1_x2, b2_x2)
    C_yy2 = np.maximum(b1_y2, b2_y2)
    C_area = (C_xx2 - C_xx1) * (C_yy2 - C_yy1)

    IOU = inter / Union
    GIOU = IOU - abs((C_area - Union) / C_area)

    return GIOU