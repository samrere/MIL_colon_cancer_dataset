import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Encoder_CC(nn.Module):
    def __init__(self):
        super(Encoder_CC, self).__init__()
        self.part1 = nn.Sequential(
            nn.Conv2d(3, 36, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(36, 48, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.part2 = nn.Sequential(
            nn.Linear(48 * 5 * 5, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout()
        )

    def forward(self, x): # D,3,27,27 --> D,512
        H = self.part1(x)
        H = H.view(-1, 48 * 5 * 5)
        H = self.part2(H)
        return H

#
class AttentionPooling(nn.Module):
    def __init__(self, attn_stru, c=1):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(512, 1)
        self.attn_stru = attn_stru
        self.c = c

        self.original_attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x): # D,E
        # attention
        if self.attn_stru == 'tanh':
            raw = self.attention(x)  # D,E -> D,1
            raw = torch.transpose(raw, -1, -2)  # 1,D
            prob = self.c * torch.tanh(raw / self.c)
            A = F.softmax(prob, -1)  # Attention weight. 1,D
            prob = (prob / self.c + 1) / 2  # [-1,1] to [0,1]
        elif self.attn_stru == 'sigmoid':
            raw = self.attention(x)  # D,E -> D,1
            raw = torch.transpose(raw, -1, -2)  # 1,D
            prob = torch.sigmoid(raw)
            A = F.softmax(prob, -1)
        elif self.attn_stru == 'none':
            raw = self.original_attention(x)  # D,E -> D,1
            raw = torch.transpose(raw, -1, -2)  # 1,D
            A = F.softmax(raw, dim=-1)
            # if A.max()==A.min():
            #     prob=A/A.max()
            # else:
            #     prob=(A-A.min())/(A.max()-A.min())
            prob = torch.sigmoid(raw)
        Vbar = A @ x  # 1,D @ D,E->1,E
        return Vbar, (prob, raw, A)

#
class Vanilla_MIL(nn.Module):
    def __init__(self, attn_stru):
        assert attn_stru in {'tanh', 'sigmoid', 'none'}
        """
        attn_stru: attention structure, choose between 'tanh', 'sigmoid', or 'none'
        """
        super(Vanilla_MIL, self).__init__()
        self.backbone = Encoder_CC()

        self.pooling = AttentionPooling(attn_stru)
        self.fc = nn.Linear(512, 1)

    def forward(self, x, labels=None, criterion=None):  # B=1,D,C,H,W
        x = x.squeeze(0) # D,C,H,W
        x = self.backbone(x)  # D,512

        Vbar, (prob, raw, A) = self.pooling(x)  # Vbar: 1,512; prob,raw,A: 1,D

        x = self.fc(Vbar) # 1,1
        x = x.flatten(0)  # 1,

        if labels is not None:
            L_bag = criterion(x, labels)
            L_inst = torch.tensor(0.).cuda()
            return x, (prob, raw, A), (L_bag, L_inst)
        else:
            return x, (prob, raw, A), None

#
class Instance_MIL(nn.Module):
    def __init__(self, attn_stru, reuse_weight, block_backprop, c):
        assert attn_stru in {'tanh', 'sigmoid', 'none'}
        """
        attn_stru: attention structure, choose between 'tanh', 'sigmoid', or 'none'
        """
        super(Instance_MIL, self).__init__()
        self.backbone = Encoder_CC()

        self.fc = nn.Linear(512, 1)
        self.pooling = AttentionPooling(attn_stru, c)
        self.reuse_weight = reuse_weight
        self.block_backprop = block_backprop

    def forward(self, x, labels=None, criterion=None, is_fully_labelled=False,full_labels=None):  # B=1,D,C,H,W
        x = x.squeeze(0)  # D,C,H,W
        D = x.shape[0]  # num of instances
        x = self.backbone(x)  # D,512

        Vbar, (prob, raw, A) = self.pooling(x)  # Vbar: 1,512; prob,raw,A: 1,D

        if self.block_backprop:
            Vbar = Vbar.detach()

        if self.reuse_weight:
            x = self.pooling.attention(Vbar) # 1,1
            if self.block_backprop:
                x = x.detach()
        else:
            x = self.fc(Vbar) # 1,1
        x = x.flatten(0)  # 1,

        if labels is not None:
            L_bag = criterion(x, labels)
            if is_fully_labelled:
                inst_labels=full_labels
            else:
                inst_labels=labels[:, None].repeat((1, D))

            L_inst = A * criterion(raw, inst_labels)
            L_inst = L_inst.sum()
            return x, (prob, raw, A), (L_bag, L_inst)
        else:
            return x, (prob, raw, A), None