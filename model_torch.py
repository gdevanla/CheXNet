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
MODEL_PATH = "checknet-batch-16-ten-crop.pth"

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
        model.train()
        running_loss = 0.0
        
        for ix, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
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
                outputs = model(images)
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
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} (Moving Avg: {moving_avg_val_loss:.4f})")
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
    train_loader, eval_loader = get_train_and_eval_dataset()
    train_model(model, train_loader,eval_loader, criterion, optimizer, scheduler, num_epochs=10)

def test_model():
    model.load_state_dict(torch.load('./chexnet_model.pth', map_location=device))
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
