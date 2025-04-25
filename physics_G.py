import os
import numpy as np
import astropy.io.fits as pyfits
#from navigator_updater.static.css import DATA_PATH
from scipy.ndimage import gaussian_filter
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from UNetauto import UNet,HybridLoss
from torchvision import transforms
import pandas as pd
# Path to training data *** do not change *** 训练数据地址
#DATA_DIR = "D:/AI4S TEEN Cup Final/Physics/train_set/"
#DATA_PATH = "D:/AI4S TEEN Cup Final/Physics/test_set/"

DATA_DIR = "/root/train_set/" #"/bohr/train-t3g1/v1"

DATA_PATH = "/root/test_set/"

DEVICE =   "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_PATH = "model.pth"

BATCH_SIZE = 4
NUM_EPOCHS = 500
LEARNING_RATE = 0.001
SMOOTHING = 3 # parameter for Gaussian Smoothing 高斯平滑参数





# Visualize halos 可视化暗物质晕



class AstroDataset(Dataset):
    def __init__(self, map_paths, cat_paths=None,transform = None):
        self.map_paths = map_paths
        self.cat_paths = cat_paths
        self.transform = transform

    def __len__(self):
        return len(self.map_paths)

    def __getitem__(self, idx):
        # Load image with gaussian smoothing
        map_data = gaussian_filter(pyfits.open(self.map_paths[idx])[0].data, sigma=SMOOTHING)
        #map_data = pyfits.open(self.map_paths[idx])[0].data
        map_data = torch.FloatTensor(map_data)
        if self.transform:
            map_data = self.transform(map_data)
        if self.cat_paths is None:
            return torch.FloatTensor(map_data).unsqueeze(0)

        # Load catalog
        cat = pyfits.open(self.cat_paths[idx])[0].data
        target = np.zeros((1024, 1024))

        for y, x in cat[:, 1:3]:
            target[int(y), int(x)] = 1.0

        return map_data.unsqueeze(0), torch.FloatTensor(target), self.cat_paths[idx]
data_size = len(os.listdir(os.path.join(DATA_DIR, 'map')))
transform = transforms.Compose([
    #transforms.Grayscale(num_output_channels=1),
    transforms.ColorJitter( brightness=0.2,    # 亮度调整有效
        contrast=0.2      # 对比度调整有效
        ),           # 禁用色调调整（灰度图像无颜色）),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    #transforms.Resize((80, 60)),
    transforms.ToTensor(),
])
# Ensure correct data and label ordering
dataset = AstroDataset([os.path.join(DATA_DIR, f'map/{i}.fits') for i in range(1, data_size+1)],
                       [os.path.join(DATA_DIR, f'cat/{i}.fits') for i in range(1, data_size+1)],transform = transform)

# Create data loader
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

"""
Define and Train Model 模型定义与训练
"""


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)



# Function for applicating model results 读取模型输出
def detect_objects(model, image_path, confidence_threshold, device=DEVICE):
    model.eval()
    with torch.no_grad():
        # Load and preprocess image
        map_data = gaussian_filter(pyfits.open(image_path)[0].data, sigma=SMOOTHING)
        image = torch.FloatTensor(map_data).unsqueeze(0).unsqueeze(0).to(device)

        # Get predictions
        output = model(image)
        predictions = output.cpu().squeeze().numpy()

        # Convert to coordinates
        coordinates = []
        for y, x in zip(*np.where(predictions > confidence_threshold)):
            confidence = predictions[y, x]
            coordinates.append((x, y, confidence))

        return coordinates


# Visualizing model predictions 可视化模型输出
def visualize_results(model, image_path, label_path, confidence_threshold, device=DEVICE):
    # Get predictions with confidence scores
    results = detect_objects(model, image_path, confidence_threshold, device)
    results = np.array(results).T if results else np.array([[], [], []])

    Z = pyfits.open(image_path)[0].data
    Z_smooth = gaussian_filter(Z, sigma=SMOOTHING)

    labels = np.transpose(pyfits.open(label_path)[0].data)

    plt.figure(figsize=(10, 10))
    plt.imshow(Z_smooth, vmin=-0.1, vmax=0.2, cmap='binary')
    plt.scatter(labels[2], labels[1], facecolors='none', edgecolors='red', s=100, label="True")
    if len(results[0]) > 0:
        # Color the scatter points based on confidence scores
        plt.scatter(results[0], results[1], facecolors='none', edgecolors='green', s=100, label="Predicted")
    plt.legend()
    plt.show()


'''
Metric function for calculating PR-AUC.
This exact function will be used for evaluation
计算PR-AUC分数
此原函数将用于比赛评测
'''


def calculate_precision_recall_curve(predictions, labels):
    print("shape of predictions: ", predictions.shape)

    # Flatten the predictions and get the indices of the sorted predictions
    flat_predictions = predictions.flatten()
    sorted_indices = np.argsort(-flat_predictions)  # Sort in descending order

    precisions = []
    recalls = []

    true_preds = 0
    num_preds = 0
    predicted_labels = 0
    num_labels = sum(len(l) for l in labels)

    labels_within_distance = [[] for _ in range(len(flat_predictions))]

    i = 0
    for image_idx, image_labels in enumerate(labels):
        for y_true, x_true in image_labels:
            for y in range(max(0, int(y_true) - 15), min(1024, int(y_true) + 16)):
                # Calculate the maximum x distance for the current y
                max_x_dist = int((max(0, 15 ** 2 - (y - y_true) ** 2)) ** 0.5)
                # Calculate the range of x-coordinates
                for x in range(max(0, int(x_true) - max_x_dist), min(1024, int(x_true) + max_x_dist + 1)):
                    coord_idx = image_idx * 1024 * 1024 + y * 1024 + x
                    labels_within_distance[coord_idx].append(i)
            i += 1

    label_predicted = [False] * num_labels

    # Iterate over sorted predictions
    for idx in sorted_indices:

        num_preds += 1

        # Determine the image index and the coordinate within the image
        image_idx = idx // (1024 * 1024)
        coord_idx = idx % (1024 * 1024)
        y, x = divmod(coord_idx, 1024)

        if len(labels_within_distance[idx]) > 0:
            true_preds += 1
            for label in labels_within_distance[idx]:
                if label_predicted[label] is False:
                    label_predicted[label] = True
                    predicted_labels += 1

        # Calculate precision and recall
        precision = true_preds / num_preds
        recall = predicted_labels / num_labels

        # Append precision and recall to the lists
        precisions.append(precision)
        recalls.append(recall)

    # Calculate PR-AUC using the trapezoidal rule
    pr_auc = np.trapz(precisions, x=recalls)

    return precisions, recalls, pr_auc


# Evaluate model 评测模型
def get_pr(model, test_loader, device=DEVICE):

    model.eval()
    for images, _, paths in test_loader:
        with torch.no_grad():
            images = images.to(device)
            outputs = model(images).cpu().numpy().squeeze(1)
            cat_data = [np.transpose(pyfits.open(path)[0].data) for path in paths]
            labels = [list(zip(cat[1], cat[2])) for cat in cat_data]
        return calculate_precision_recall_curve(outputs, labels)


# Model instance training 模型训练
torch.manual_seed(42)
model = UNet().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()

#class_weights = torch.tensor([1.0, 5.0])
#criterion = HybridLoss(weight=class_weights).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses = []
TEST_DATA_DIR = DATA_PATH + 'halo_testA'
test_size = len(os.listdir(os.path.join(TEST_DATA_DIR, 'map')))
test_dataset = AstroDataset([os.path.join(TEST_DATA_DIR, f'map/{i}.fits') for i in range(1, test_size + 1)],
                            [os.path.join(TEST_DATA_DIR, f'cat/{i}.fits') for i in range(1, test_size+1)])
test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=False) #test_loader = DataLoader(dataset, batch_size=test_size, shuffle=False)

scores = []
if os.path.exists("model.pth"):
    load = torch.load("model.pth")
    start_epoch = load["epoch"] + 1
    model.load_state_dict (load["model"])
    optimizer.load_state_dict(load["optimizer"])

else:
    start_epoch = 0
df = {"epoches":[],"scores":[]}

for epoch in range(start_epoch,NUM_EPOCHS):
    model.train()
    total_loss = 0

    for images, targets, _ in dataloader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)
        #targets = targets.long()

        optimizer.zero_grad()
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / len(dataloader)
    train_losses.append(train_loss)

    print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Training Loss: {train_loss:.4f}')
    if epoch % 5 == 0:


        torch.save({"epoch":epoch,
                    "model":model.state_dict(),
                    "optimizer":optimizer.state_dict()},f"model{epoch}.pth")
        model.eval()
        pre,rec,sc = get_pr(model,test_loader)
        print(f"PR-AUC{sc}")
        df["epoches"].append(epoch)
        df["scores"].append(sc)
        scores.append(sc)
        df_save = pd.DataFrame(df)
        df_save.to_csv(f"score{start_epoch}.csv",index = False)


# Plot training curve 训练过程可视化
plt.figure(figsize=(12, 4))

plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.title('Training Loss')
plt.grid(True)
plt.plot(scores, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('scores')
plt.yscale('log')
plt.title('Training scores')
plt.grid(True)
plt.show()

visualize_results(model, dataset.map_paths[0], dataset.cat_paths[0], confidence_threshold=0.08)
# Calculate PR-AUC 计算PR-AUC分数
precisions, recalls, pr_auc = get_pr(model, dataloader)
print(f"PR-AUC Score: {pr_auc:.4f}")

# Plot PR curve 可视化精确率-召回率曲线
plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve (AUC = {pr_auc:.4f})')
plt.grid(True)
plt.show()

"""
Submission 提交结果
"""
import zipfile



if os.environ.get('DATA_PATH') == None:###这里改了的！！！！！！！！！！！！！！！！！！！！！！！！！




    # Submit testA for public leaderboard 提交选手公开傍结果
    #DATA_PATH = os.environ.get('DATA_PATH') + "/"



    TEST_DATA_DIR = DATA_PATH + 'halo_testA'
    test_size = len(os.listdir(os.path.join(TEST_DATA_DIR, 'map')))
    dataset = AstroDataset([os.path.join(TEST_DATA_DIR, f'map/{i}.fits') for i in range(1, test_size + 1)])
    loader = DataLoader(dataset, batch_size=test_size, shuffle=False)
    model.eval()
    for images in loader:
        with torch.no_grad():
            outputs = model(images.to(DEVICE)).cpu().numpy().squeeze(1)
        np.save('submissionsA.npy', outputs)
    # Submit testB for private leaderboard 提交评测数据结果
    TEST_DATA_DIR = DATA_PATH + 'halo_testB'
    test_size = len(os.listdir(os.path.join(TEST_DATA_DIR, 'map')))
    dataset = AstroDataset([os.path.join(TEST_DATA_DIR, f'map/{i}.fits') for i in range(1, test_size + 1)])
    loader = DataLoader(dataset, batch_size=test_size, shuffle=False)
    model.eval()
    for images in loader:
        with torch.no_grad():
            outputs = model(images.to(DEVICE)).cpu().numpy().squeeze(1)
        np.save('submissionsB.npy', outputs)

    # The final submission will be a zip file containing the your model outputs for both testing sets
    # 最终提交一个压缩文件包括两个npy文件。
    with zipfile.ZipFile('submission.zip', 'w') as zipf:
        zipf.write('submissionsA.npy')
        zipf.write('submissionsB.npy')
