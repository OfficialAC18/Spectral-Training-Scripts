#Author - Akchunya Chanchal
import os
import tempfile
from pathlib import Path
from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
from tqdm import tqdm
from ray import train, tune
from ray.train import Checkpoint, RunConfig
from ray.tune.schedulers import ASHAScheduler
import numpy as np
from insilico_lrcn import InSilicoLRCN
from torch.utils.data import DataLoader, Dataset

#For calculating F1-Score over a loop
#For calculating F1-Score over a loop
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix

class F1_score_running:
    def __init__(self,classes = None):
        self.tp = [] if not classes else [0]*classes
        self.fp = [] if not classes else [0]*classes
        self.fn = [] if not classes else [0]*classes
        self.counts = [] if not classes else [0]*classes
        self.classes = classes
        
    def log(self, predicted, labels):
        #Get the total number of classes that exist from the data
        if not self.classes:
            self.classes = len(np.unique(labels))
            self.fp = [0] * self.classes
            self.fn = [0] * self.classes
            self.tp = [0] * self.classes
            self.counts = [0] * self.classes
        
        #We need to calculate the FP and FN for each class 
        for idx in range(self.classes):
            #In a copy of the predicted and label arrays
            #Change all the indices with val = i to 1, and everything else to 0
            pred_idx = np.array(list(map(lambda x: 1 if x == idx else 0, predicted)))
            label_idx = np.array(list(map(lambda x: 1 if x == idx else 0, labels)))
            
            #Count the number of 1s and -1s and the support for the class
            unique, counts = np.unique(pred_idx - label_idx, return_counts = True)
            counts = dict(zip(unique, counts))
            self.fp[idx] += counts[1] if 1 in counts.keys() else 0
            self.fn[idx] += counts[-1] if -1 in counts.keys() else 0
            self.tp[idx] += np.logical_and(pred_idx, label_idx).sum().item()
            self.counts[idx] += np.sum(len(np.argwhere(label_idx)))
    
    def calc(self,average = None):
        #Convert to numpy array for vectorisation
        self.tp = np.array(self.tp,dtype=np.float32)
        self.fp = np.array(self.fp,dtype=np.float32)
        self.fn = np.array(self.fn,dtype=np.float32)
        ratio = np.array(self.counts)/sum(self.counts)

        #Calculate Precision and Recall
        self.precision = self.tp/(self.tp + self.fp)
        self.recall = self.tp/(self.tp + self.fn)
        
        #Calculate the F1-Score
        f1 = 2*self.tp
        f1 /= 2*self.tp + self.fn + self.fp

        if not average:
            return f1
        elif average == 'binary':
            return f1[-1]
        elif average == 'macro':
            return np.mean(f1)
        elif average == 'weighted':
            return sum(ratio * f1)
        return f1


#Training Script for in-vivo models
class ExVivoDataset(Dataset):
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

        spec_path = os.path.join(f"{self.file_dir}/{self.files[idx].stem}.npy")

        #Attain Label
        if 'class0' in self.files[idx].stem:
            label = 0
        elif 'class1' in self.files[idx].stem:
            label = 1
        elif 'class2' in self.files[idx].stem:
            label = 2
        
        #Load file
        spectra = torch.Tensor(np.load(spec_path),device=None).expand(1,-1)
        return spectra, label
    

def train_spectra(config,dataset_dir):
    #Train, Val, Test Datasets
    train_dset = ExVivoDataset(f"{dataset_dir}/Train")
    val_dset = ExVivoDataset(f"{dataset_dir}/Val")

    #Create DataLoaders
    train_dataloader = DataLoader(
        train_dset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
    )

    val_dataloader = DataLoader(
        val_dset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
    )

    
    #Get the size of the input
    for inp, _ in train_dataloader:
        input_shape = inp.shape
        break
    
    #Create a random tensor, to initialize the model
    init_input = torch.rand(size = input_shape)
    # print("Init shape:",init_input.shape)

    model = InSilicoLRCN(
        config['nc1'],
        config['k1'],
        config['nc2'],
        config['k2'],
        config['nc3'],
        config['k3'],
        config['nc4'],
        config['k4'],
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

    best_val_acc = 0.0
    best_val_f1 = 0.0
    #Running Training Epochs
    for epoch in range(100):
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(train_dataloader):
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
            if i%10 == 0:
                print('loss: %.3f'%(running_loss/epoch_steps))
        scheduler.step()

       # Validation loss
        val_loss = 0.0
        val_steps = 0
        val_f1 = F1_score_running()
        total = 0
        correct = 0
        for i, data in tqdm(enumerate(val_dataloader),f"Val Epoch:{epoch}"):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                predicted = torch.argmax(outputs.data, -1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = loss_fn(outputs, labels)
                val_f1.log(predicted.cpu().numpy(), labels.cpu().numpy())
                val_loss += loss.cpu().numpy()
                val_steps += 1
        
        val_acc = correct / total
        val_f1 = val_f1.calc(average = "weighted")
        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be accessed through in ``get_checkpoint()``
        # in future iterations.
        # Note to save a file like checkpoint, you still need to put it under a directory
        # to construct a checkpoint.
        train.report(
        {"loss": (val_loss / val_steps),
        "accuracy": val_acc,
        "f1": val_f1,
        "best_accuracy": best_val_acc,})

        if val_f1 > best_val_f1:
            #Update the best loss
            best_val_f1 = val_f1
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
                torch.save(
                    (model.state_dict(), optimizer.state_dict()), path
                )
                print(path)
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                train.report(
                    {"loss": (val_loss / val_steps),
                    "accuracy": val_acc,
                    "f1": val_f1,
                    "best_accuracy": best_val_acc,},
                    checkpoint=checkpoint,
                )
    print("Finished Training")

def test_best_model(best_config,dataset_dir):
    test_dset = ExVivoDataset(f"{dataset_dir}/Test")

    test_dataloader = DataLoader(
        test_dset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
    )
    
    
    for inp, _ in test_dataloader:
        input_shape = inp.shape
        break
        
    #Create a random tensor, to initialize the model
    init_input = torch.rand(size = input_shape)
    
    best_model = InSilicoLRCN(
        best_config.config['nc1'],
        best_config.config['k1'],
        best_config.config['nc2'],
        best_config.config['k2'],
        best_config.config['nc3'],
        best_config.config['k3'],
        best_config.config['nc4'],
        best_config.config['k4'],
        best_config.config['dp'],
        best_config.config['l1'],
        best_config.config['l2'],
        init_input
    )

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    

    checkpoint_path = os.path.join(best_config.checkpoint.to_directory(), "checkpoint.pt")
    model_state, _ = torch.load(checkpoint_path)
    best_model.load_state_dict(model_state)

    best_model.to(device)


    #Testing
    test_loss = 0.0
    test_steps = 0
    total = 0
    f1 = F1_score_running()
    correct = 0
    with torch.no_grad():
        for data in tqdm(test_dataloader,"Test Epoch"):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = best_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            f1.log(torch.argmax(outputs,-1).cpu().numpy(),labels.cpu().numpy())
            correct += (predicted == labels).sum().item()

    print("Best Test Set Accuracy:{}".format(correct/total))
    print("Best F1-Score:",f1.calc(average = "weighted"))

def main(dataset_dir, num_samples=10, max_num_epochs=500, gpus_per_trial=0.5,dset_name = None):
    config = {
        'nc1': tune.sample_from(lambda: 2 ** np.random.randint(2,9)),
        'k1': tune.choice([1,3,5]),
        'nc2':tune.sample_from(lambda: 2 ** np.random.randint(2,9)),
        'k2':tune.choice([1,3,5]),
        'nc3':tune.sample_from(lambda: 2 ** np.random.randint(2,9)),
        'k3':tune.choice([1,3,5]),
        'nc4':tune.sample_from(lambda: 2 ** np.random.randint(2,9)),
        'k4':tune.choice([1,3,5]),
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
            resources={"cpu": 24, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
        run_config=RunConfig(storage_path=f"/scratch/users/k23058970/Spectral-Training-Scripts/results/{dset_name}")
    )
    results = tuner.fit()
    
    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))
    print("Best trial f1-score: {}".format(
        best_result.metrics['f1']
    ))

    test_best_model(best_result,dataset_dir=dataset_dir)