import os
import zipfile
import gdown
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.utils.data as data
from numpy import interp
from itertools import cycle
from tqdm.notebook import tqdm
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize

def load_data(drive_path, zip_path, extract_path, file_id):
    # Create directories if needed
    os.makedirs('./drive/MyDrive', exist_ok=True)

    # Download the file if it doesn't exist
    if not os.path.exists(drive_path):
        if not os.path.exists(zip_path):
            print("Downloading dataset...")
            gdown.download(id=file_id, output=zip_path, quiet=False)
            print("Download complete.")

        # Unzip the file
        print("Extracting zip...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Extraction complete.")

        # Rename the folder if needed
        default_folder = os.path.join(extract_path, 'dataset')
        if os.path.exists(default_folder):
            os.rename(default_folder, drive_path)
            print(f"Renamed folder to {drive_path}")

def display_data():
    # Number of samples to display per class
    n = 5

    # Plot the samples with no substructure
    i = 1
    print('Samples with no substructure: ')
    plt.rcParams['figure.figsize'] = [14, 14]  # Set the figure size
    for image in train_files1[:n]:
        ax = plt.subplot(3, n, i)  # Create subplot
        plt.imshow(np.load(image).reshape(64, 64), cmap='gray')  # Load and display the image
        ax.get_xaxis().set_visible(False)  # Hide x-axis
        ax.get_yaxis().set_visible(False)  # Hide y-axis
        i += 1
    plt.show()  # Show the plot

    # Plot the samples with spherical substructure
    print('Samples with spherical substructure: ')
    i=1
    plt.rcParams['figure.figsize'] = [14, 14]  # Set the figure size
    for image in train_files2[:n]:
        ax = plt.subplot(3, n, i)  # Create subplot
        plt.imshow(np.load(image).reshape(64, 64), cmap='gray')  # Load and display the image
        ax.get_xaxis().set_visible(False)  # Hide x-axis
        ax.get_yaxis().set_visible(False)  # Hide y-axis
        i += 1
    plt.show()  # Show the plot

    # Plot the samples with vortex substructure
    print('Samples with vortex substructure: ')
    i=1
    plt.rcParams['figure.figsize'] = [14, 14]  # Set the figure size
    for image in train_files3[:n]:
        ax = plt.subplot(3, n, i)  # Create subplot
        plt.imshow(np.load(image).reshape(64, 64), cmap='gray')  # Load and display the image
        ax.get_xaxis().set_visible(False)  # Hide x-axis
        ax.get_yaxis().set_visible(False)  # Hide y-axis
        i += 1
    plt.show()

    print('Samples with no substructure in fourier space: ')
    i=1
    plt.rcParams['figure.figsize'] = [14, 14]  # Set the figure size
    for image in train_files1[:n]:
        ax = plt.subplot(3, n, i)  # Create subplot
        fft_image =  np.fft.fft(np.load(image))
        fft_real = fft_image.imag
        plt.imshow(fft_real.reshape(64, 64), cmap='gray')  # Load and display the image
        ax.get_xaxis().set_visible(False)  # Hide x-axis
        ax.get_yaxis().set_visible(False)  # Hide y-axis
        i += 1
    plt.show()

    print('Samples with no substructure in SVD space: ')
    i=1
    plt.rcParams['figure.figsize'] = [14, 14]  # Set the figure size
    for image in train_files1[:n]:
        ax = plt.subplot(3, n, i)  # Create subplot
        U, S, V = np.linalg.svd(np.load(image))
        plt.imshow(U.reshape(64,64), cmap='gray')
        ax.get_xaxis().set_visible(False)  # Hide x-axis
        ax.get_yaxis().set_visible(False)  # Hide y-axis
        i += 1
    plt.show()
    print(np.cumsum(S.reshape(64))/np.sum(S))

def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result

# Define the Convolutional Neural Network (CNN) class
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Define the first convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=2, padding=0)

        # Define the second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=0)

        # Define the third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=3, stride=1, padding=0)

        # Define the first fully connected (linear) layer
        self.linear1 = nn.Linear(120, 64)

        # Define the second fully connected (linear) layer
        self.linear2 = nn.Linear(64, 3)

        # Define the activation function
        self.tanh = nn.Tanh()

        # Define the average pooling layer
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Apply the first convolutional layer
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.avgpool(x)

        # Apply the second convolutional layer
        x = self.conv2(x)
        x = self.tanh(x)
        x = self.avgpool(x)

        # Apply the third convolutional layer
        x = self.conv3(x)
        x = self.tanh(x)

        # Flatten the tensor for the fully connected layers
        x = x.reshape(x.shape[0], -1)

        # Apply the first fully connected layer
        x = self.linear1(x)
        x = self.tanh(x)

        # Apply the second fully connected layer
        x = self.linear2(x)

        return x

# Define the Vision Transformer class
class VIT(nn.Module):
    def __init__(self, chw=(1, 64, 64), patch_size = 8, data_rank=124*2, num_classes=3):
        super(VIT, self).__init__()
        patch_vec_size = patch_size**2
        sequence_length = int(chw[1] * chw[2] / patch_vec_size) + 1

        self.patchify = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, data_rank))

        self.pos_embedding = get_positional_embeddings(sequence_length, data_rank)

        self.patch_embedding = nn.Linear(patch_vec_size, data_rank)

        self.layer_norm = nn.LayerNorm(data_rank)

        self.multi_attention = nn.MultiheadAttention(data_rank, num_heads=1, kdim=data_rank, vdim=data_rank)

        self.mlp_first = nn.Linear(data_rank, 2*data_rank)
        self.mlp_second = nn.Linear(2*data_rank, data_rank)
        self.relu = nn.ReLU()

        self.classifier = nn.Linear(data_rank, num_classes)


    def forward(self, x):
        B = x.shape[0]

        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = self.patchify(x)

        x = self.patch_embedding(x)

        x = torch.cat((cls_tokens, x), dim=1)

        pos_emb = self.pos_embedding.to(x.device)

        x = pos_emb + x

        x = x.transpose(0, 1)

        x_tmp = self.layer_norm(x)

        x_tmp, _ = self.multi_attention(x_tmp, x_tmp, x_tmp)

        x = x + x_tmp

        x = self.layer_norm(x)

        x_tmp = self.mlp_first(x)

        x_tmp = self.relu(x_tmp)

        x_tmp = self.mlp_second(x_tmp)

        x = x + x_tmp

        x = x.transpose(0, 1) 

        cls_output = x[:, 0, :] 

        x = self.classifier(cls_output)

        return x

def train(model, train_data_loader):
    # Loss Function
    criteria = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Calculate the number of batches for training data
    n_batches_train = (len(train_files1) * 3) / batch_size  # Equal number of files in each class

    # Set the number of training epochs
    n_epochs = 20
    loss_array = []  # To store the loss values

    print(len(train_data_loader))

    # Progress bar for epochs
    pbar = tqdm(range(1, n_epochs + 1))
    for epoch in pbar:
        train_loss = 0.0
        train_acc = 0.0

        # Iterate over the training data loader
        for step, (x_tr, y_tr) in enumerate(train_data_loader):
            data = x_tr.to(device).float()  # Move input data to the device and convert to float
            labels = y_tr.to(device, dtype=torch.long)  # Move labels to the device and convert to long
            optimizer.zero_grad()  # Clear the gradients
            outputs = model(data)  # Forward pass through the model
            _, preds = torch.max(outputs.data, 1)  # Get the predictions
            correct = (preds == labels).float().sum()  # Calculate the number of correct predictions
            loss = criteria(outputs, labels)  # Calculate the loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update the model parameters

            train_loss += loss.item()  # Accumulate the loss
            train_acc += correct.item() / data.shape[0]  # Accumulate the accuracy

        # Calculate the average loss and accuracy for the epoch
        train_loss = train_loss / n_batches_train
        train_acc = train_acc / n_batches_train

        # Display the training statistics
        pbar.set_postfix({'Training Loss': train_loss, 'Training Acc': train_acc})

def test(model, val_data_loader):
    # Initialize lists to store scores and labels
    y_score = []
    y_test = []

    # Iterate over the validation data loader
    for _, (x_ts, y_ts) in enumerate(val_data_loader):
        mini_val_data = x_ts.to(device).float()  # Move validation data to the device and convert to float
        y_ts = y_ts.to(device, dtype=torch.long)  # Move labels to the device and convert to long

        with torch.no_grad():  # Disable gradient calculation for validation
            outputs = model(mini_val_data)  # Forward pass through the model
            probabilities = torch.nn.functional.softmax(outputs, dim=1)  # Apply softmax to get probabilities

        # Append the probabilities and labels to the respective lists
        y_score.append(probabilities.cpu().detach().numpy())
        y_test.append(y_ts.cpu().detach().numpy())

    # Convert the lists to numpy arrays and reshape them
    y_score = np.asarray(y_score).reshape(-1, 3)
    y_val = np.asarray(y_test).reshape(-1)

    # Binarize the labels for multi-class evaluation
    y_val = label_binarize(y_val, classes=[0, 1, 2])


    # Convert the predicted probabilities to class labels
    y_pred = np.argmax(y_score, axis=1)

    # Convert y_val from one-hot encoding to class labels
    y_true = np.argmax(y_val, axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy}')

    # Calculate precision
    precision = precision_score(y_true, y_pred, average='weighted')
    print(f'Precision: {precision}')

    # Calculate recall
    recall = recall_score(y_true, y_pred, average='weighted')
    print(f'Recall: {recall}')

    # Calculate F1 score
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f'F1 Score: {f1}')


    # Define a function to load .npy files

def npy_loader(path):
    sample = torch.from_numpy(np.load(path))  # Load the numpy file and convert it to a torch tensor
    return sample



if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Optional, useful for frozen executables

    # Paths
    drive_path = './drive/MyDrive/lensing_dataset'
    zip_path = './drive/MyDrive/dataset.zip'
    extract_path = './drive/MyDrive/'

    # Google Drive file ID
    file_id = '1AZAJzJdm6FJT4rIyY9N_6FcKLf8VZtn8'

    # Create directories if needed
    os.makedirs('./drive/MyDrive', exist_ok=True)

    # Define the input paths
    train_path1 = drive_path + '/train/no'
    train_files1 = [os.path.join(train_path1, f) for f in os.listdir(train_path1) if f.endswith(".npy")]
    train_path2 = drive_path + '/train/cdm'
    train_files2 = [os.path.join(train_path2, f) for f in os.listdir(train_path2) if f.endswith(".npy")]
    train_path3 = drive_path + '/train/axion'
    train_files3 = [os.path.join(train_path3, f) for f in os.listdir(train_path3) if f.endswith(".npy")]

    # Set Batch Size
    batch_size = 150

    # Load training data
    train_data = torchvision.datasets.DatasetFolder(root=drive_path+'/train', loader=npy_loader, extensions='.npy')
    print("Training Classes: " + str(train_data.class_to_idx))  # Print the classes found in the training data
    train_data_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)  # Create a data loader for training data

    # Load validation data
    val_data = torchvision.datasets.DatasetFolder(root=drive_path+'/val', loader=npy_loader, extensions='.npy')
    print("Validation Classes: " + str(val_data.class_to_idx))  # Print the classes found in the validation data
    val_data_loader = data.DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)  # Create a data loader for validation data



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Instantiate the CNN model and move it to the appropriate device
    model = VIT().to(device)

    train(model, train_data_loader)

    test(val_data_loader)
