**We will upload the complete code once the paper has been published.**
# FedKb

This project implements the Federated learning with Knowledge Bank (FedKb) algorithm using the LeNet-5, ResNet-18 model on the MNIST and BloodMNIST dataset with a Dirichlet distribution (alpha=0.1 and 0.2 respectively) for non-IID data partitioning.

## Project Structure
fedkb/
├── data/                  # Directory for MNIST data (auto-downloaded)
├── src/                   # Source code
│   ├── init.py        # Module initialization
│   ├── model.py           # LeNet-5 model definition
│   ├── server.py          # Server logic with FedAvg
│   ├── client.py          # Client local training logic
│   ├── utils.py           # Utility functions (data split, evaluation)
│   ├── config.py          # Configuration file with adjustable parameters
├── run.py                 # Main script to run the program
├── requirements.txt       # Required Python libraries
└── README.md              # Project documentation

## Prerequisites

- Python 3.8 or higher
- Git (for cloning the repository)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/fedkb.git
   cd fedavg-mnist
Install dependencies:
bash

pip install -r requirements.txt
Usage
Run the main script to start the federated learning process:

python run.py
Configuration
All adjustable parameters are defined in src/config.py. You can modify:

GLOBAL_SEED: Global seed for reproducibility (default: 42)
NUM_CLIENTS: Total number of clients (default: 10)
NUM_ROUNDS: Number of training rounds (default: 3)
NUM_CLIENTS_PER_ROUND: Number of clients selected per round (default: 3)
LOCAL_EPOCHS: Number of local epochs per client (default: 5)
DATA_DIR: Directory for MNIST data (default: './data')
ALPHA: Dirichlet parameter for non-IID data (default: 1.0)
SERVER_PORT: Server port (default: 9999)
BUFFER_SIZE: Socket buffer size (default: 4096)
LEARNING_RATE: Learning rate for SGD (default: 0.01)
BATCH_SIZE: Batch size for DataLoader (default: 32)
The program will:

Download the MNIST dataset to DATA_DIR.
Split the data using Dirichlet distribution (alpha=ALPHA).
Run FedAvg with LeNet-5 for NUM_ROUNDS rounds, selecting NUM_CLIENTS_PER_ROUND clients randomly each round.
Evaluate the global model on the test set after each round.
Expected Output

Server started...

--- Round 1/3 ---
Selected clients: [8, 1, 5]
Starting clients for round 1
Client 8 generated: 50
Client 1 generated: 37
Client 5 generated: 47
Connection from ('127.0.0.1', <port1>)
Connection from ('127.0.0.1', <port2>)
Connection from ('127.0.0.1', <port3>)
Round 1 completed. Global model updated.
Global model - Accuracy: 85.12%, Loss: 0.4610
...
Notes
If you encounter Address already in use, stop the program and rerun, or change SERVER_PORT in src/config.py.
The MNIST dataset will be downloaded automatically on the first run.
License
This project is licensed under the MIT License.

---

### Hướng dẫn áp dụng
1. **Tạo file `config.py`**:
   - Tạo file `D:\mylife\FEDKB_GIT\src\config.py` và sao chép nội dung từ `src/config.py` ở trên.

2. **Cập nhật các file khác**:
   - Thay thế nội dung của `server.py`, `client.py`, `utils.py`, và `run.py` trong thư mục `D:\mylife\FEDKB_GIT\src\` và `D:\mylife\FEDKB_GIT\` bằng các phiên bản đã cập nhật ở trên.

3. **Chạy lại**:
   ```cmd
   cd D:\mylife\FEDKB_GIT
   python run.py
Thay đổi tham số:
Nếu muốn thay đổi số client, số vòng, hoặc bất kỳ tham số nào khác, chỉ cần chỉnh sửa trong D:\mylife\FEDKB_GIT\src\config.py.
