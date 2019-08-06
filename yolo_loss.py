# -*- coding:utf-8 -*-
__author__ = 'Leo.Z'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device('cuda')


class YoloLoss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        """
            Constructor..
        :param S: cell = SxS = 14x14
        :param B: box = 2
        :param l_coord: lambda-coord = 5
        :param l_noobj: lambda-noobj = 0.5
        """
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def compute_iou(self, box1, box2):
        """
            Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        """
        # N = 2
        N = box1.size(0)
        # M = 1
        M = box2.size(0)

        lt = torch.max(
            box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh < 0] = 0  # clip at 0
        # 交集部分
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]
        # I/U
        iou = inter / (area1 + area2 - inter)
        return iou  # [N,M] = [2,1]

    def forward(self, pred_tensor, target_tensor):
        """
            Compute loss..
        :param pred_tensor: pred (tensor) size= Batchsize,S,S,Bx5+20=30  5-->[x,y,w,h,c]
        :param target_tensor: label (tensor) size= Batchsize,S,S,Bx5+20=30  5-->[x,y,w,h,c]
        :return: loss
        """
        pred_tensor = pred_tensor.to(device)
        target_tensor = target_tensor.to(device)

        # N = BatchSize
        N = pred_tensor.size()[0]
        # coo_mask 表示在target中那些cell有object (C>0表示有obj)
        coo_mask = target_tensor[:, :, :, 4] > 0
        # print("coo_mask1:", coo_mask.size())  # [b,14,14] cell数量，有obj的为true
        # noo_mask 表示在target中那些cell没有有object (C==0表示没有obj)
        noo_mask = target_tensor[:, :, :, 4] == 0
        # print("noo_mask1:", noo_mask.size())  # [b,14,14] cell数量，无obj为true

        # 将coo_mask的维度调整和target一致
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)
        # print("coo_mask2:", coo_mask.size())  # [b, 14, 14, 30]
        # 将noo_mask的维度调整和target一致
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)
        # print("noo_mask2:", noo_mask.size())  # [b, 14, 14, 30]

        # 从pred_tensor中过滤出应该有obj的cell
        coo_pred = pred_tensor[coo_mask].view(-1, 30)
        # print("coo_pred:", coo_pred.size())  # [COO_T, 30]  3136个cell中有obj的数量为COO_T
        # 取其中box部分数据，并将其按box展开，数量是2倍有obj的cell数量
        box_pred = coo_pred[:, :10].contiguous().view(-1, 5)
        # print("box_pred:", box_pred.size())  # [COO_T*2, 5]  3136个cell中有obj的cell中box的数量为COO_T*2
        # 另外取剩下classes部分，数量是有obj的cell数量
        class_pred = coo_pred[:, 10:]
        # print("class_pred:", class_pred.size())  # [COO_T, 20]

        # 对target做同样的处理
        coo_target = target_tensor[coo_mask].view(-1, 30)  # [COO_T, 30]  3136个cell中有obj的数量为COO_T
        box_target = coo_target[:, :10].contiguous().view(-1, 5)  # [COO_T*2, 5]  3136个cell中有obj的cell中box的数量为COO_T*2
        class_target = coo_target[:, 10:]  # [COO_T, 20]

        # 从pred_tensor中过滤不应该有obj部分cell
        noo_pred = pred_tensor[noo_mask].view(-1, 30)
        # print("noo_pred:", noo_pred.size())  # [NOO_T, 30]  3136个cell中无obj的数量为NOO_T
        # 从target_tensor中过滤不含obj部分cell
        noo_target = target_tensor[noo_mask].view(-1, 30)
        # print("noo_target:", noo_target.size())  # [NOO_T, 30]  3136个cell中无obj的数量为NOO_T

        ###===================================论文第4项损失，不含obj的confidence损失===========================
        ### 创建过滤器，准备过滤不含obj的pred中的confidence，并计算其损失，论文第4项
        noo_pred_mask = torch.ByteTensor(noo_pred.size()).to(device)
        # print("noo_pred_mask:", noo_pred_mask.size())  # [NOO_T, 30]
        noo_pred_mask.zero_()
        noo_pred_mask[:, 4] = 1
        noo_pred_mask[:, 9] = 1
        noo_pred_c = noo_pred[noo_pred_mask]  # noo pred只需要计算 c 的损失 size[-1,2]
        # print("noo_pred_c:", noo_pred_c.size())  # [NOO_T*2]
        noo_target_c = noo_target[noo_pred_mask]
        # print("noo_target_c:", noo_target_c.size())  # [NOO_T*2]
        nooobj_loss = F.mse_loss(noo_pred_c, noo_target_c, reduction='sum')
        ###===================================论文第4项损失，不含obj的confidence损失===========================

        # 计算了应该包含obj 部分cell的confidence的损失
        coo_response_mask = torch.ByteTensor(box_target.size()).to(device)
        # print("coo_response_mask:", coo_response_mask.size())  # [COO_T*2, 5]
        coo_response_mask.zero_()
        coo_not_response_mask = torch.ByteTensor(box_target.size()).to(device)
        # print("coo_not_response_mask:", coo_not_response_mask.size())  # [COO_T*2, 5]
        coo_not_response_mask.zero_()

        box_target_iou = torch.zeros(box_target.size()).to(device)
        # print("box_target_iou:", box_target_iou.size())  # [COO_T*2, 5]

        # 对于每个target的box，选择一个与之IoU最大的pred box，供后面计算box的loss
        for i in range(0, box_target.size()[0], 2):  # range(0, COO_T*2, 2):
            box1 = box_pred[i:i + 2]
            box1_xyxy = torch.FloatTensor(box1.size()).to(device)
            # x,y为什么除以14，这里使用的是14x14 grid。这里获取框的左上点和右下点
            box1_xyxy[:, :2] = box1[:, :2] / 14. - 0.5 * box1[:, 2:4]
            box1_xyxy[:, 2:4] = box1[:, :2] / 14. + 0.5 * box1[:, 2:4]
            box2 = box_target[i].view(-1, 5)
            box2_xyxy = torch.FloatTensor(box2.size()).to(device)
            box2_xyxy[:, :2] = box2[:, :2] / 14. - 0.5 * box2[:, 2:4]
            box2_xyxy[:, 2:4] = box2[:, :2] / 14. + 0.5 * box2[:, 2:4]
            iou = self.compute_iou(box1_xyxy[:, :4], box2_xyxy[:, :4])  # [2,4] vs [1,4]
            max_iou, max_index = iou.max(0)
            max_index = max_index.data.cuda()

            # 在mask中标记2个框中，哪个框对应IoU更大，大的为response，小的为not responce
            coo_response_mask[i + max_index] = 1
            coo_not_response_mask[i + 1 - max_index] = 1

            #####
            # we want the confidence score to equal the
            # intersection over union (IOU) between the predicted box
            # and the ground truth
            #####
            box_target_iou[i + max_index, torch.LongTensor([4]).cuda()] = (max_iou).data.cuda()
        box_target_iou = Variable(box_target_iou).cuda()
        # print("box_target_iou:", box_target_iou.size())  # [COO_T*2, 5]
        # 1.response loss
        box_pred_response = box_pred[coo_response_mask].view(-1, 5)
        # print("box_pred_response:", box_pred_response.size())  # [COO_T, 5]
        box_target_response_iou = box_target_iou[coo_response_mask].view(-1, 5)
        # print("box_target_response_iou:", box_target_response_iou.size())  # [COO_T, 5]
        box_target_response = box_target[coo_response_mask].view(-1, 5)
        # print("box_target_response:", box_target_response.size())  # [COO_T, 5]

        # 计算匹配上最大IoU的部分的损失(response)
        contain_loss = F.mse_loss(box_pred_response[:, 4], box_target_response_iou[:, 4], reduction='sum')
        # 顺便计算坐标损失
        loc_loss = F.mse_loss(box_pred_response[:, :2], box_target_response[:, :2], reduction='sum') + F.mse_loss(
            torch.sqrt(box_pred_response[:, 2:4]), torch.sqrt(box_target_response[:, 2:4]), reduction='sum')

        # 2.not response loss
        box_pred_not_response = box_pred[coo_not_response_mask].view(-1, 5)
        # print("box_pred_not_response:", box_target_response.size())  # [COO_T, 5]
        box_target_not_response = box_target[coo_not_response_mask].view(-1, 5)
        # print("box_target_not_response:", box_target_response.size())  # [COO_T, 5]
        box_target_not_response[:, 4] = 0
        # not_contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response[:,4],size_average=False)

        # 计算未匹配上的IoU部分损失(not response)
        not_contain_loss = F.mse_loss(box_pred_not_response[:, 4], box_target_not_response[:, 4], reduction='sum')

        # 计算所有含obj 部分cell的分类损失
        class_loss = F.mse_loss(class_pred, class_target, reduction='sum')

        # 论文第1,2项 有obj的中心点和w,h的loss * lambda=5
        return (self.l_coord * loc_loss +
                # 论文第3项 所有框中，响应的和不响应的。响应的乘以2?
                2 * contain_loss +
                not_contain_loss +
                # 论文第4项 * lambda=0.5
                self.l_noobj * nooobj_loss +
                # 论文第5项，20类误差
                class_loss) / N


# if __name__ == '__main__':
#     a = torch.rand(64, 14, 14, 30)
#     b = torch.rand(64, 14, 14, 30)
#     loss = YoloLoss(14, 2, 5, 0.5)
#     loss.forward(a, b)
