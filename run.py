import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import threading
import time
from src.model import LeNet5, OCTTransFormer, ResNet18NoBatchNorm, DermaMNISTNet, BloodMNISTNet, ResNet18Blood
from src.server import start_server
from src.client import start_client
from src.utils import non_iid_partition_dirichlet
from src.config import GLOBAL_SEED, NUM_ROUNDS, NUM_CLIENTS, DATA_DIR, BATCH_SIZE
from src.config import DEVICE

def load_dataset(dataset_name):
    if dataset_name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
        
    elif dataset_name.lower() == 'octmnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1904,), (0.2085,))
        ])
        try:
            import medmnist
            from medmnist import OCTMNIST
            train_dataset = OCTMNIST(split='train', root=DATA_DIR, download=True, transform=transform)
            test_dataset = OCTMNIST(split='test', root=DATA_DIR, download=True, transform=transform)
        except ImportError:
            raise ImportError("Please install medmnist: pip install medmnist")
            
    elif dataset_name.lower() == 'dermamnist':
        transform = transforms.Compose([
            transforms.RandomCrop(28, padding=1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        try:
            import medmnist
            from medmnist import DermaMNIST
            train_split = DermaMNIST(split='train', root=DATA_DIR, download=True, transform=transform)
            test_split = DermaMNIST(split='test', root=DATA_DIR, download=True, transform=transform)
            train_dataset = ConcatDataset([train_split, test_split])
            # Combine labels from train and test splits
            train_labels = np.concatenate([train_split.labels, test_split.labels], axis=0)
            train_dataset.labels = train_labels  # Attach labels to ConcatDataset
            val_dataset = None
            test_dataset = DermaMNIST(split='val', root=DATA_DIR, download=True, transform=transform_test)
        except ImportError:
            raise ImportError("Please install medmnist: pip install medmnist")

    elif dataset_name.lower() == 'bloodmnist':
        transform = transforms.Compose([
            transforms.RandomCrop(28, padding=1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        try:
            import medmnist
            from medmnist import BloodMNIST
            train_split = BloodMNIST(split='train', root=DATA_DIR, download=True, transform=transform)
            test_split = BloodMNIST(split='test', root=DATA_DIR, download=True, transform=transform)
            train_dataset = ConcatDataset([train_split, test_split])
            # Combine labels from train and test splits
            train_labels = np.concatenate([train_split.labels, test_split.labels], axis=0)
            train_dataset.labels = train_labels  # Attach labels to ConcatDataset
            val_dataset = None
            test_dataset = BloodMNIST(split='val', root=DATA_DIR, download=True, transform=transform_test)
        except ImportError:
            raise ImportError("Please install medmnist: pip install medmnist")
            
    elif dataset_name.lower() == 'pathmnist':
        transform = transforms.Compose([
            transforms.RandomCrop(28, padding=1),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.7405, 0.5330, 0.7056), (0.1230, 0.1518, 0.1192))
        ])
        transform_full = transforms.Compose([
            transforms.RandomCrop(28, padding=1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            transforms.ToTensor(),
            transforms.Normalize((0.7405, 0.5330, 0.7056), (0.1230, 0.1518, 0.1192))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.7405, 0.5330, 0.7056), (0.1230, 0.1518, 0.1192))
        ])
        try:
            import medmnist
            from medmnist import PathMNIST
            train_dataset = PathMNIST(split='train', root=DATA_DIR, download=True, transform=transform_full)
            test_dataset = PathMNIST(split='test', root=DATA_DIR, download=True, transform=transform_test)
        except ImportError:
            raise ImportError("Please install medmnist: pip install medmnist")
            
    else:
        raise ValueError("Dataset must be 'mnist', 'octmnist', 'dermamnist', 'bloodmnist', or 'pathmnist'")

    return train_dataset, test_dataset

def run_server(global_model, selected_clients_list, algorithm, proportions, test_loader, global_control, model_name):
    print("Starting server...")
    global_control = start_server(global_model, selected_clients_list, algorithm=algorithm, proportions=proportions, test_loader=test_loader, global_control=global_control, model_name=model_name)
    print("Server finished.")
    return global_control

def run_clients(global_model, selected_clients, algorithm, client_datasets, global_control, model_name):
    client_threads = []
    for client_id in selected_clients:
        print(f"Starting client {client_id}...")
        seed = GLOBAL_SEED + int(client_id)
        t = threading.Thread(target=start_client, args=(client_id, seed, client_datasets[client_id], global_model, algorithm, global_control, model_name))
        client_threads.append(t)
        t.start()
    
    for t in client_threads:
        t.join()
    print("All clients for this round finished.")

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    
    while True:
        algorithm = input("Enter the federated learning algorithm (fedavg, fedprox, scaffold, selfdistillcore): ").strip().lower()
        if algorithm in ['fedavg', 'fedprox', 'scaffold', 'selfdistillcore']:
            break
        print("Invalid input! Please enter 'fedavg', 'fedprox', 'scaffold', or 'selfdistillcore'.")

    while True:
        dataset_name = input("Enter the dataset (mnist, octmnist, dermamnist, bloodmnist, pathmnist): ").strip().lower()
        if dataset_name in ['mnist', 'octmnist', 'dermamnist', 'bloodmnist', 'pathmnist']:
            break
        print("Invalid input! Please enter 'mnist', 'octmnist', 'dermamnist', 'bloodmnist', or 'pathmnist'.")

    # Automatically select the model based on the dataset
    if dataset_name.lower() == 'mnist':
        model_name = 'lenet5'
    elif dataset_name.lower() == 'octmnist':
        model_name = 'octtransformer'
    elif dataset_name.lower() == 'dermamnist':
        model_name = 'dermamnistnet'
    elif dataset_name.lower() == 'bloodmnist':
        model_name = 'resnet18blood'        
    elif dataset_name.lower() == 'pathmnist':
        model_name = 'resnet18nobatchnorm'

    print(f"Running with algorithm: {algorithm}, dataset: {dataset_name}, model: {model_name}")

    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    train_dataset, test_dataset = load_dataset(dataset_name)
    client_datasets, proportions = non_iid_partition_dirichlet(train_dataset, NUM_CLIENTS, partition="hetero")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model initialization
    if model_name == 'lenet5':
        global_model = LeNet5().to(DEVICE)
    elif model_name == 'octtransformer':
        global_model = OCTTransFormer().to(DEVICE)
    elif model_name == 'dermamnistnet':
        global_model = DermaMNISTNet().to(DEVICE)
    elif model_name == 'resnet18blood':
        global_model = ResNet18Blood().to(DEVICE)        
    elif model_name == 'resnet18nobatchnorm':
        global_model = ResNet18NoBatchNorm().to(DEVICE)
    global_control = None

    selected_clients_list = []
    for round_num in range(NUM_ROUNDS):
        np.random.seed(GLOBAL_SEED + round_num)
        selected_clients = np.random.choice(NUM_CLIENTS, np.random.randint(10, 16), replace=False)
        selected_clients_list.append(selected_clients)

    server_thread = threading.Thread(target=run_server, args=(global_model, selected_clients_list, algorithm, proportions, test_loader, global_control, model_name))
    server_thread.daemon = True
    server_thread.start()

    time.sleep(2)

    for round_num in range(NUM_ROUNDS):
        print(f"\nStarting clients for round {round_num+1}")
        run_clients(global_model, selected_clients_list[round_num], algorithm, client_datasets, global_control, model_name)
        time.sleep(2)

    server_thread.join()
    print("Training completed.")