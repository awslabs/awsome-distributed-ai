# Modified version of https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu_torchrun.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import mlflow
import mlflow.pytorch

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def ddp_setup():
    # Use NCCL backend for GPU training, fallback to GLOO for CPU
    if torch.cuda.is_available():
        print("Using NCCL backend for GPU training")
        init_process_group(backend="nccl")
    else:
        print("Using GLOO backend for CPU training") 
        init_process_group(backend="gloo")

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
        use_mlflow: bool = False,
        tracking_uri: str = None
    ) -> None:
        self.model = model
        self.rank = int(os.environ["RANK"])
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        self.use_mlflow = use_mlflow
        self.tracking_uri = tracking_uri if tracking_uri else f"file://{os.environ['HOME']}/mlruns"
        # Set device
        self.device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.device.index] if torch.cuda.is_available() else None)

    def _load_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path, map_location=self.device)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
        source = source.to(self.device)
        targets = targets.to(self.device)
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        self.train_data.sampler.set_epoch(epoch)
        total_loss = 0
        for source, targets in self.train_data:
            loss = self._run_batch(source, targets)
            total_loss += loss
        
        avg_loss = total_loss / len(self.train_data)
        if self.use_mlflow and self.rank == 0:  # Only log from rank 0
            try:
                mlflow.log_metric("train_loss", avg_loss, step=epoch)
            except Exception as e:
                print(f"Warning: MLflow metric logging failed: {e}")
        print(f"[RANK {self.rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)} | Loss: {avg_loss}")
        return avg_loss

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        if self.use_mlflow and self.rank == 0:
            try:
                print(f"Setting tracking URI to {self.tracking_uri}")
                # Set tracking URI first
                if self.tracking_uri:
                    mlflow.set_tracking_uri(self.tracking_uri)

                # Create or get experiment
                experiment = mlflow.get_experiment_by_name("mnist_ddp")
                if experiment is None:
                    experiment_id = mlflow.create_experiment("mnist_ddp")
                else:
                    experiment_id = experiment.experiment_id

                # Set the experiment
                mlflow.set_experiment(experiment_id=experiment_id)

                with mlflow.start_run():
                    mlflow.log_params({
                        "model": "MLP",
                        "optimizer": "Adam",
                        "learning_rate": self.optimizer.param_groups[0]['lr'],
                        "batch_size": len(next(iter(self.train_data))[0]),
                        "epochs": max_epochs,
                        "device": str(self.device)
                    })
                    mlflow.pytorch.log_model(self.model.module, "model")
            except Exception as e:
                print(f"Warning: MLflow initialization failed, continuing without tracking: {e}")
                self.use_mlflow = False
        else:
            print("MLFlow is disabled")

        for epoch in range(self.epochs_run, max_epochs):
            avg_loss = self._run_epoch(epoch)
            if epoch % self.save_every == 0:
                self._save_snapshot(epoch)
                if self.use_mlflow and self.rank == 0:
                    try:
                        mlflow.pytorch.log_model(self.model.module, f"model_epoch_{epoch}")
                        mlflow.log_metric("train_loss", avg_loss, step=epoch)
                    except Exception as e:
                        print(f"Warning: MLflow logging failed: {e}")

def load_train_objs():
    # Define data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST dataset
    train_set = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # Create model and optimizer
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    return train_set, model, optimizer

def prepare_dataloader(dataset: datasets.MNIST, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

def main(save_every: int, total_epochs: int, batch_size: int, snapshot_path: str, use_mlflow: bool = False, tracking_uri: str = None):
    ddp_setup()
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path, use_mlflow, tracking_uri)
    trainer.train(total_epochs)
    destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--checkpoint_path', default="./snapshot.pt", type=str, help='Full path to checkpoint file')
    parser.add_argument('--use_mlflow', action='store_true', help='Enable MLFlow logging')
    parser.add_argument('--tracking_uri', type=str, help='MLflow tracking URI', default=None)
    args = parser.parse_args()
    main(args.save_every, args.total_epochs, args.batch_size, args.checkpoint_path, args.use_mlflow, args.tracking_uri)
