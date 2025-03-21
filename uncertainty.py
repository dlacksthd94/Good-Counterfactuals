import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import pickle
import utils
from tqdm import tqdm

# ------------------------------
# Model Definition
# ------------------------------

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
torch.seed(1234)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ------------------------------
# Step 1: Data Splitting
# ------------------------------

def load_and_split_mnist(train_ratio=0.8):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_size = int(train_ratio * len(dataset))
    calib_size = len(dataset) - train_size
    train_set, calib_set = random_split(dataset, [train_size, calib_size])
    return train_set, calib_set

# ------------------------------
# Step 2: Model Training
# ------------------------------

def train_model(train_set, epochs=5, batch_size=64, lr=0.001):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
    return model

def test_model(model, test_set, batch_size=64):
    """
    Compute nonconformity scores using the chosen method: 'brier', 'margin', or 'hinge'.

    Parameters:
        model: Trained PyTorch model
        test_set: Calibration dataset
        method: Nonconformity score type ('brier', 'margin', 'hinge')
        batch_size: Batch size for processing

    Returns:
        Array of nonconformity scores
    """
    model.eval()
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    preds = []
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = torch.softmax(model(images), dim=1)
            outputs = outputs.detach().cpu()
            pred = outputs.argmax(dim=1)
            correct += (labels == pred).sum().item()
            total += len(labels)
            preds.append(pred)
    
    preds = torch.concat(preds)
    acc = correct / total
    return acc, preds

# ------------------------------
# Step 3: Compute Nonconformity Scores
# ------------------------------

def compute_calibration_scores(model, calib_set, method, batch_size=64):
    """
    Compute nonconformity scores using the chosen method: 'brier', 'margin', or 'hinge'.

    Parameters:
        model: Trained PyTorch model
        calib_set: Calibration dataset
        method: Nonconformity score type ('brier', 'margin', 'hinge')
        batch_size: Batch size for processing

    Returns:
        Array of nonconformity scores
    """
    model.eval()
    calib_loader = DataLoader(calib_set, batch_size=batch_size, shuffle=False)
    scores = []
    
    with torch.no_grad():
        for images, labels in calib_loader:
            images = images.to(device)
            outputs = torch.softmax(model(images), dim=1)
            outputs = outputs.detach().cpu()
            true_probs = outputs.gather(1, labels.unsqueeze(1)).squeeze()

            if method == "brier":
                scores.extend((1 - true_probs) ** 2)  # Squared error of true label probability

            elif method == "margin":
                sorted_probs, _ = torch.sort(outputs, descending=True)
                margins = sorted_probs[:, 0] - sorted_probs[:, 1]  # Difference between top-1 and top-2
                scores.extend(1 - margins)  # Lower margin = more uncertainty

            elif method == "hinge":
                max_wrong_probs, _ = torch.max(outputs.scatter(1, labels.unsqueeze(1), -1), dim=1)
                hinge_scores = max_wrong_probs - true_probs  # Difference between best incorrect and true
                scores.extend(hinge_scores)

            else:
                raise ValueError("Invalid method. Choose from 'brier', 'margin', 'hinge'.")

    return np.array(scores)

# ------------------------------
# Step 4: Select Quantile Threshold
# ------------------------------

def get_quantile_threshold(calibration_scores, alpha=0.1):
    n = len(calibration_scores)
    k = int(np.ceil((1 - alpha) * (n + 1))) - 1
    sorted_scores = np.sort(calibration_scores)
    threshold = sorted_scores[min(k, n - 1)]
    return threshold

# ------------------------------
# Step 5: Conformal Prediction Function
# ------------------------------

def conformal_predict(model, x, threshold, method="brier"):
    model.eval()
    x = x.to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x.unsqueeze(0)), dim=1).squeeze()

    prediction_set = []
    for y in range(10):
        if method == "brier":
            score = (1 - probs[y].item()) ** 2  # Brier score
        elif method == "margin":
            sorted_probs = torch.sort(probs, descending=True).values
            score = 1 - (sorted_probs[0] - sorted_probs[1]).item()  # Margin score
        elif method == "hinge":
            max_wrong_prob = max(probs[:y].max(), probs[y+1:].max()).item()
            score = max_wrong_prob - probs[y].item()  # Hinge score
        else:
            raise ValueError("Invalid method. Choose from 'brier', 'margin', 'hinge'.")

        if score <= threshold:
            prediction_set.append(y)

    return prediction_set
# ------------------------------
# Additional: Size of Prediction Set for a Specific Image
# ------------------------------

def prediction_set_size(model, x, threshold, method="brier"):
    prediction_set = conformal_predict(model, x, threshold, method)
    return len(prediction_set)

# ------------------------------
# Example Usage
# ------------------------------

if __name__ == "__main__":
    # Step 1: Load and split MNIST
    train_set, calib_set = load_and_split_mnist(train_ratio=0.8)

    # Step 2: Train model
    model = train_model(train_set, epochs=3)  # Quick training

    # Choose nonconformity method: 'brier', 'margin', or 'hinge'
    method = "brier"
    # method = "margin"
    # method = "hing642653e"

    # Step 3: Compute calibration scores
    calib_scores = compute_calibration_scores(model, calib_set, method)

    # Step 4: Get quantile threshold
    alpha = 0.1  # 90% coverage
    threshold = get_quantile_threshold(calib_scores, alpha)
    print(f"Threshold for alpha={alpha} using {method}: {threshold:.4f}")

    # Step 5 & Additional: Predict and get set size for a test image
    mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())
    acc, preds = test_model(model, mnist_test)

    x_test, y_test = mnist_test[0]  # Take first test image and label
    
    pred_set = conformal_predict(model, x_test, threshold, method)
    print(f"Prediction set: {pred_set}, True label: {y_test}")

    set_size = prediction_set_size(model, x_test, threshold, method)
    print(f"Prediction set size: {set_size}")

file_name = 'test_black_patch_w8h8.pickle'
with open('data/' + file_name, 'rb') as f:
    test_black_patch = pickle.load(f)

alpha = 0.01  # 99% coverage
threshold = get_quantile_threshold(calib_scores, alpha)
print(f"Threshold for alpha={alpha} using {method}: {threshold:.4f}")

test_black_patch_uncertain = []
for image_original, images_perturbed, label in tqdm(test_black_patch):
    list_pred_set = []
    list_set_size = []
    for image_perturbed in images_perturbed:
        pred_set = conformal_predict(model, image_perturbed, threshold, method)
        set_size = prediction_set_size(model, image_perturbed, threshold, method)
        list_pred_set.append(pred_set)
        list_set_size.append(set_size)
    instance_uncertain = (image_original, images_perturbed, label, list_pred_set)
    list_uncertain_pred_set = np.where(np.array(list_set_size) > 1)[0]
    if list_uncertain_pred_set.size:
        test_black_patch_uncertain.append(instance_uncertain)

with open('data/' + file_name.split('.')[0] + '_uncertain' + '.pickle', 'wb') as f:
    pickle.dump(test_black_patch_uncertain, f)

