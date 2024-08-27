import os
import tempfile
from pathlib import Path
from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
from torcheval.metrics.functional import multiclass_f1_score as f1
from tqdm import tqdm
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
import numpy as np
from insilico_conv import InSilicoConv
from torch.utils.data import DataLoader, Dataset

#Training Script for in-silico models
class InSilicoDataset(Dataset):
    def __init__(self,
                 file_dir,
                 transform=None,
                 target_transform=None):
        
        self.file_dir = file_dir
        self.files = list(Path(self.file_dir).rglob("*.npy"))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        #Attain File Path
        if 'negative' in self.files[idx].stem:
            spec_path = os.path.join(f"{self.file_dir}/Negative/{self.files[idx].stem}.npy")
            label = 0
        else:
            spec_path = os.path.join(f"{self.file_dir}/Positive/{self.files[idx].stem}.npy")
            label = 1
        
        #Load file
        spectra = torch.Tensor(np.load(spec_path),device=None).expand(1,-1)

        return spectra, label
    

def train_spectra(config,dataset_dir):
    #Train, Val, Test Datasets
    train_dset = InSilicoDataset(f"{dataset_dir}/Train")
    val_dset = InSilicoDataset(f"{dataset_dir}/Val")

    #Create DataLoaders
    train_dataloader = DataLoader(
        train_dset,
        batch_size=512,
        shuffle=True,
        num_workers=8,
    )

    val_dataloader = DataLoader(
        val_dset,
        batch_size=512,
        shuffle=True,
        num_workers=8,
    )

    #Get the size of the input
    for inp, _ in train_dataloader:
        input_shape = inp.shape
        # print("Input shape:",input_shape)
        break
    
    #Create a random tensor, to initialize the model
    init_input = torch.rand(size = input_shape)
    # print("Init shape:",init_input.shape)

    model = InSilicoConv(
        config['nc1'],
        config['k1'],
        config['nc2'],
        config['k2'],
        config['nc3'],
        config['k3'],
        config['nc4'],
        config['k4'],
        config['nc5'],
        config['k5'],
        config['dp'],
        config['l1'],
        config['l2'],
        init_input
    )

    #Set device
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    #Put model on device (GPU/CPU)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=config['lr'],
                          momentum=0.1,
                          nesterov=True,
                          dampening=0)
    scheduler = OneCycleLR(optimizer=optimizer,
                           max_lr = config['lr'],
                           epochs=100,
                           steps_per_epoch=len(train_dataloader))
    
    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

    best_val_acc= 0.0
    #Running Training Epochs
    for epoch in range(100):
        running_loss = 0.0
        epoch_steps = 0
        for i, data in tqdm(enumerate(train_dataloader),f"Train Epoch:{epoch}/100"):
            inputs,labels = data
            inputs,labels = inputs.to(device), labels.to(device)

            #Zero the parameter gradients
            optimizer.zero_grad()
            #Forward + Backward + Optmize
            outputs = model(inputs)
            loss = loss_fn(outputs,labels)
            loss.backward()
            optimizer.step()

            #Get Some Stats
            running_loss += loss.item()
            epoch_steps +=1
            if i%200 == 0:
                print('loss: %.3f'%(running_loss/epoch_steps))
        scheduler.step()

       # Validation loss
        val_loss = 0.0
        val_steps = 0
        avg_f1 = []
        total = 0
        correct = 0
        for i, data in tqdm(enumerate(val_dataloader),f"Val Epoch:{epoch}/20"):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                predicted = torch.argmax(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = loss_fn(outputs, labels)
                avg_f1.append(f1(predicted,labels).cpu().numpy())
                val_loss += loss.cpu().numpy()
                val_steps += 1
        
        val_acc = correct / total
        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be accessed through in ``get_checkpoint()``
        # in future iterations.
        # Note to save a file like checkpoint, you still need to put it under a directory
        # to construct a checkpoint.
        train.report(
        {"loss": (val_loss / val_steps),
        "accuracy": val_acc,
        "best_accuracy": best_val_acc,
        "avg_f1": np.mean(avg_f1)})

        if val_acc > best_val_acc:
            #Update the best loss
            best_val_acc = val_acc
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
                torch.save(
                    (model.state_dict(), optimizer.state_dict()), path
                )
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                train.report(
                    {"loss": (val_loss / val_steps),
                    "accuracy": val_acc,
                    "best_accuracy": best_val_acc,
                    "avg_f1": np.mean(avg_f1),},
                    checkpoint=checkpoint,
                )
    print("Finished Training")

def test_best_model(best_config,dataset_dir):
    best_model = InSilicoConv(
        best_config.config['nc1'],
        best_config.config['k1'],
        best_config.config['nc2'],
        best_config.config['k2'],
        best_config.config['nc3'],
        best_config.config['k3'],
        best_config.config['nc4'],
        best_config.config['k4'],
        best_config.config['nc5'],
        best_config.config['k5'],
        best_config.config['dp'],
        best_config.config['l1'],
        best_config.config['l2']
    )

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    
    best_model.to(device)

    checkpoint_path = os.path.join(best_config.checkpoint.to_directory(), "checkpoint.pt")

    model_state, _ = torch.load(checkpoint_path)
    best_model.load_state_dict(model_state)
    test_dset = InSilicoDataset(f"{dataset_dir}/Test")

    test_dataloader = DataLoader(
        test_dset,
        batch_size=512,
        shuffle=True,
        num_workers=8,
    )

    #Testing
    test_loss = 0.0
    test_steps = 0
    avg_f1 = []
    total = 0
    correct = 0
    with torch.no_grad():
        for data in tqdm(test_dataloader,"Test Epoch"):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = best_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            avg_f1.append(f1(predicted,labels))
            correct += (predicted == labels).sum().item()

    print("Best Test Set Accuracy:{}".format(correct/total))
    print("Best F1-Score:",np.mean(avg_f1))

def main(dataset_dir, num_samples=50, max_num_epochs=100, gpus_per_trial=1):
    config = {
        'nc1': tune.sample_from(lambda: 2 ** np.random.randint(2,9)),
        'k1': tune.choice([1,3,5,7]),
        'nc2':tune.sample_from(lambda: 2 ** np.random.randint(2,9)),
        'k2':tune.choice([1,3,5,7]),
        'nc3':tune.sample_from(lambda: 2 ** np.random.randint(2,9)),
        'k3':tune.choice([1,3,5,7]),
        'nc4':tune.sample_from(lambda: 2 ** np.random.randint(2,9)),
        'k4':tune.choice([1,3,5,7]),
        'nc5':tune.sample_from(lambda: 2 ** np.random.randint(2,9)),
        'k5':tune.choice([1,3,5]),
        'dp':tune.choice([0.1,0.2,0.25,0.50]),
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 15)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 15)),
        "lr": tune.loguniform(1e-3, 1e-1),
    }
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=15,
        reduction_factor=2)
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(partial(train_spectra,dataset_dir=dataset_dir)),
            resources={"cpu": 8, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
    )
    results = tuner.fit()
    
    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))
    print("Best trail avg f1-score: {}".format(
        best_result.metrics['avg_f1']
    ))

    test_best_model(best_result,dataset_dir=dataset_dir)