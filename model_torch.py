import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from read_data import ChestXrayDataSet
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet18, DenseNet121_Weights



N_CLASSES = 14
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
#DATA_DIR = './ChestX-ray14/images'
DATA_DIR = '/home/guru/data/chexray-14/images/'

TEST_IMAGE_LIST = './ChestX-ray14/labels/test_list.txt'
VAL_IMAGE_LIST = './ChestX-ray14/labels/val_list.txt'
TRAIN_IMAGE_LIST = './ChestX-ray14/labels/train_list.txt'
BATCH_SIZE = 8
#MODEL_PATH = "chestxnet-batch-16-ten-crop.pth"
MODEL_PATH = "chexnet_model.pth"

normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])


transform = transforms.Compose([
    transforms.Resize((224, 224)),
#    transforms.Resize(256),
    transforms.TenCrop(224),
    transforms.Lambda
    (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    transforms.Lambda
    (lambda crops: torch.stack([normalize(crop) for crop in crops]))
])

# Define image transformations
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

def get_train_eval_dataset():
    # Load data (replace with actual paths and labels)
    train_dataset = ChestXrayDataSet(
        DATA_DIR,
        TRAIN_IMAGE_LIST,
        transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


    eval_dataset = ChestXrayDataSet(
        DATA_DIR,
        VAL_IMAGE_LIST,
        transform=transform)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, eval_loader
    
#train_loader = eval_loader

# Load pre-trained DenseNet121 model
model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
num_ftrs = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Linear(num_ftrs, len(CLASS_NAMES)),
    nn.Sigmoid()
)


# model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
# num_ftrs = model.fc.in_features
# model.fc = nn.Sequential(
#     nn.Linear(num_ftrs, len(CLASS_NAMES)),
#     nn.Sigmoid()
# )

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True)

# Training loop with Improved Early Stopping
def train_model(model, train_loader, eval_loader, criterion, optimizer, scheduler, num_epochs=10, save_path=MODEL_PATH, patience=5, min_delta=0.001):
    best_val_loss = float("inf")
    epochs_no_improve = 0
    val_loss_history = []
    
    for epoch in range(num_epochs):

        start_time = time.time()
        
        model.train()
        running_loss = 0.0
        
        for ix, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # dealing with tencrop
            batch_size, num_crops, c, h, w = images.shape
            images = images.view(batch_size * num_crops, c, h, w)
            
            optimizer.zero_grad()
            outputs = model(images)

            # Reshape outputs and average over 10 crops: [batch_size, 14]
            outputs = outputs.view(batch_size, num_crops, -1).mean(dim=1)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        
        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_outputs = []
        with torch.no_grad():
            for images, labels in eval_loader:
                images, labels = images.to(device), labels.to(device)

                # dealing with tencrop
                batch_size, num_crops, c, h, w = images.shape
                images = images.view(batch_size * num_crops, c, h, w)
            
                outputs = model(images)
                # Reshape outputs and average over 10 crops: [batch_size, 14]
                outputs = outputs.view(batch_size, num_crops, -1).mean(dim=1)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                all_labels.append(labels.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())
        
        avg_val_loss = val_loss / len(eval_loader)
        val_loss_history.append(avg_val_loss)
        
        # Compute moving average of validation loss
        if len(val_loss_history) > 3:
            moving_avg_val_loss = np.mean(val_loss_history[-3:])
        else:
            moving_avg_val_loss = avg_val_loss
        
        all_labels = np.vstack(all_labels)
        all_outputs = np.vstack(all_outputs)
        all_preds = (all_outputs > 0.5).astype(int)
        
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        auc = roc_auc_score(all_labels, all_outputs, average='macro')

        end_time = time.time()
        epoch_time = end_time - start_time
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} (Moving Avg: {moving_avg_val_loss:.4f}), Time: {epoch_time:.2f} sec")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, AUC: {auc:.4f}")
        
        # Adjust learning rate based on validation loss
        scheduler.step(avg_val_loss)
        
        # Early stopping logic with min_delta
        if moving_avg_val_loss < best_val_loss - min_delta:
            best_val_loss = moving_avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch+1} with validation loss {moving_avg_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")
            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break
    
    print("Training complete")



# Train the model
def train():
    train_loader, eval_loader = get_train_eval_dataset()
    train_model(model, train_loader,eval_loader, criterion, optimizer, scheduler, num_epochs=10)

def test_model():
    model.load_state_dict(torch.load(MODEL_PATH)) #map_location=device)
    print(f'Loading {MODEL_PATH=}')
    model.eval()

    
    test_dataset = ChestXrayDataSet(
        DATA_DIR,
        TEST_IMAGE_LIST,
        transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()


    for i, (inp, target) in enumerate(test_loader):
        target = target.cuda()
        gt = torch.cat((gt, target), 0)
        bs, n_crops, c, h, w = inp.size()
        input_var = torch.autograd.Variable(inp.view(-1, c, h, w).cuda(), volatile=True)
        output = model(input_var)
        output_mean = output.view(bs, n_crops, -1).mean(1)
        pred = torch.cat((pred, output_mean.data), 0)
        if i % 500 == 0:
            print(f'processing {i}')

    import pandas as pd
    pd.DataFrame(gt.cpu().numpy()).to_csv('./gt.txt')
    pd.DataFrame(pred.cpu().numpy()).to_csv('./pred.txt')

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

if __name__ == '__main__':
    #train()
    test_model()


"""
Epoch [1/10], Train Loss: 0.1821, Val Loss: 0.1956 (Moving Avg: 0.1956), Time: 5027.46 sec
Precision: 0.0730, Recall: 0.0023, F1-score: 0.0036, AUC: 0.6572
Model saved at epoch 1 with validation loss 0.1956
Epoch [2/10], Train Loss: 0.1767, Val Loss: 0.1752 (Moving Avg: 0.1752), Time: 5033.50 sec
Precision: 0.0633, Recall: 0.0163, F1-score: 0.0214, AUC: 0.6693
Model saved at epoch 2 with validation loss 0.1752
Epoch [3/10], Train Loss: 0.1727, Val Loss: 0.4925 (Moving Avg: 0.4925), Time: 5050.51 sec
Precision: 0.0639, Recall: 0.0166, F1-score: 0.0201, AUC: 0.6913
No improvement for 1 epoch(s)
Epoch [4/10], Train Loss: 0.1683, Val Loss: 0.1642 (Moving Avg: 0.2773), Time: 4969.05 sec
Precision: 0.0944, Recall: 0.0193, F1-score: 0.0273, AUC: 0.7406
No improvement for 2 epoch(s)
Epoch [5/10], Train Loss: 0.1646, Val Loss: 0.1606 (Moving Avg: 0.2724), Time: 5008.31 sec
Precision: 0.1221, Recall: 0.0093, F1-score: 0.0158, AUC: 0.7615
No improvement for 3 epoch(s)
Epoch [6/10], Train Loss: 0.1618, Val Loss: 0.1580 (Moving Avg: 0.1609), Time: 4984.97 sec
Precision: 0.1682, Recall: 0.0109, F1-score: 0.0192, AUC: 0.7724
Model saved at epoch 6 with validation loss 0.1609
Epoch [7/10], Train Loss: 0.1595, Val Loss: 0.1580 (Moving Avg: 0.1589), Time: 5014.43 sec
Precision: 0.1690, Recall: 0.0234, F1-score: 0.0360, AUC: 0.7784
Model saved at epoch 7 with validation loss 0.1589
Epoch [8/10], Train Loss: 0.1578, Val Loss: 0.1564 (Moving Avg: 0.1575), Time: 4904.01 sec
Precision: 0.1352, Recall: 0.0185, F1-score: 0.0286, AUC: 0.7802
Model saved at epoch 8 with validation loss 0.1575
Epoch [9/10], Train Loss: 0.1562, Val Loss: 0.1559 (Moving Avg: 0.1568), Time: 4999.03 sec
Precision: 0.1937, Recall: 0.0305, F1-score: 0.0449, AUC: 0.7911
No improvement for 1 epoch(s)
Epoch [10/10], Train Loss: 0.1548, Val Loss: 0.1562 (Moving Avg: 0.1562), Time: 4950.61 sec
Precision: 0.1762, Recall: 0.0311, F1-score: 0.0441, AUC: 0.7836
Model saved at epoch 10 with validation loss 0.1562
Training complete
"""

"""
chexnet-batch-16-ten-crop using densenet latest weights
The AUROC of Atelectasis is 0.780081748516357
The AUROC of Cardiomegaly is 0.8867306315925055
The AUROC of Effusion is 0.8615192872473203
The AUROC of Infiltration is 0.6794907714615743
The AUROC of Mass is 0.7682471439404299
The AUROC of Nodule is 0.662111501773603
The AUROC of Pneumonia is 0.7119206617529034
The AUROC of Pneumothorax is 0.8228817323431947
The AUROC of Consolidation is 0.7861354990226994
The AUROC of Edema is 0.8744510823310527
The AUROC of Emphysema is 0.8350242971881072
The AUROC of Fibrosis is 0.7435034498157753
The AUROC of Pleural_Thickening is 0.7488050843764947
The AUROC of Hernia is 0.8609624189991303
"""

"""
chexnet-model.pth (no ten fold crop, older densenet model)
The average AUROC is 0.812
The AUROC of Atelectasis is 0.7973307742529339
The AUROC of Cardiomegaly is 0.894666957923871
The AUROC of Effusion is 0.8713408208278822
The AUROC of Infiltration is 0.6980316012000554
The AUROC of Mass is 0.7728915919760989
The AUROC of Nodule is 0.7401608083269692
The AUROC of Pneumonia is 0.7531693289402188
The AUROC of Pneumothorax is 0.8371376252300848
The AUROC of Consolidation is 0.7935737066362432
The AUROC of Edema is 0.8799135938493071
The AUROC of Emphysema is 0.8863208103435731
The AUROC of Fibrosis is 0.7997239446477478
The AUROC of Pleural_Thickening is 0.7476994193406368
The AUROC of Hernia is 0.8942432227234156
"""


