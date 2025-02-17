# encoding: utf-8

"""
The main CheXNet model implementation.
"""

import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from read_data import ChestXrayDataSet
from sklearn.metrics import roc_auc_score


CKPT_PATH = 'model.pth.tar'
N_CLASSES = 14
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
DATA_DIR = './ChestX-ray14/images'
TEST_IMAGE_LIST = './ChestX-ray14/labels/test_list.txt'
BATCH_SIZE = 8


norm_pattern = r'norm\.(\d+)\.'
norm_replacement = r'norm\1.'

conv_pattern = r'conv\.(\d+)\.'
conv_replacement = r'conv\1.'


def load_model():
    #cudnn.benchmark = True

    # initialize and load the model
    model = DenseNet121(N_CLASSES) # .cuda()
    #model = torch.nn.DataParallel(model) #.cuda()

    if os.path.isfile(CKPT_PATH):
        print("=> loading checkpoint")
        checkpoint = torch.load(CKPT_PATH, map_location=torch.device('cpu'))

        # Remove 'module.' prefix from state_dict keys
        checkpoint_state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for key, value in checkpoint_state_dict.items():
            new_key = key.replace('module.', '')  # Remove 'module.' from each key
            # Applying the regex substitution
            new_key = re.sub(norm_pattern, norm_replacement, new_key)
            new_key = re.sub(conv_pattern, conv_replacement, new_key)

            new_state_dict[new_key] = value
        #model.load_state_dict(new_state_dict)

        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

        # Report missing/unexpected keys
        print(f"Missing keys: {missing_keys[:5]}")
        print(f"Unexpected keys: {unexpected_keys[:5]}")
        if missing_keys or unexpected_keys:
            print("Fix missing key/unexpected key error")
            return None

        print("=> loaded checkpoint")
    else:
        print("=> no checkpoint found")
    # switch to evaluate mode
    model.eval()
    print('Returning model')
    return model

def prepare_data():

    return test_dataset, test_loader

normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

def lambda1(crops):
    return torch.stack([transforms.ToTensor()(crop) for crop in crops])

def lambda2(crops):
    return torch.stack([normalize(crop) for crop in crops])

def main():

    model = load_model()

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    test_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                    image_list_file=TEST_IMAGE_LIST,
                                    transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.TenCrop(224),
                                        transforms.Lambda(lambda1),
                                        transforms.Lambda(lambda2)])
                                    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0) # pin_memory=True)



    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    #gt = gt.cuda()
    pred = torch.FloatTensor()
    #pred = pred.cuda()

    for i, (inp, target) in enumerate(test_loader):
        if (i > 10000):
            break
        #target = target.cuda()
        gt = torch.cat((gt, target), 0)
        bs, n_crops, c, h, w = inp.size()
        input_var = torch.autograd.Variable(
            inp.view(-1, c, h, w), volatile=True)
        output = model(input_var)
        output_mean = output.view(bs, n_crops, -1).mean(1)
        pred = torch.cat((pred, output_mean.data), 0)
        print(f'processing {i}')


    #return (gt, pred)
    AUROCs = compute_AUCs(gt, pred)
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs[i]))


def compute_AUCs(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.

    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.

    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs


class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


if __name__ == '__main__':
    main()
