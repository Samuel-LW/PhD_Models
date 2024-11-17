import torch
import torch.nn as nn
import torch.optim
import numpy as np
import torchvision.transforms as transforms
import pandas as pd


from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, ExponentialLR
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from datetime import datetime
from monai.transforms import RandRotate90, Resize, RandFlip
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from dataloader import RandomFmriDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from pytorchtools import EarlyStopping

'''
The original file was based on this project and uses its dataloader (any small changes are mentoned in the papers/thesis) https://github.com/Vishal232406/FMRI_CNN .

I based the train test on this https://github.com/Sifat-Ahmed/Pytorch-Binary-Classification/blob/main/main.py and also sk-learn documentation/examples.

I based the model on https://github.com/xmuyzz/3D-CNN-PyTorch/blob/master/models/cnn.py and https://github.com/Vishal232406/FMRI_CNN however most has changed.

The average pooling idea comes from this paper https://arxiv.org/abs/1312.4400 https://towardsdatascience.com/transfer-learning-from-pre-trained-models-f2393f124751 https://d2l.ai/chapter_convolutional-modern/nin.html
but did not end up in the final model

The train val test is spaghetti code, I would clean it up if I had more time.

pytorchtool is an early stopping script widely avaliable online
'''

class CNN_model(nn.Module):
    def __init__(self, Dropout):
        super(CNN_model, self).__init__()
        self.convolutional_layer = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(6, 6, 6), stride=1, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=(6, 6, 6), stride=2),
            nn.BatchNorm3d(16),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(6, 6, 6), stride=1, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=(6, 6, 6), stride=2),
            nn.BatchNorm3d(32),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(4, 4, 4), stride=1, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=(4, 4, 4), stride=2),
            nn.BatchNorm3d(64))
            
        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=64, out_features=20),
            nn.LeakyReLU(),
            nn.BatchNorm1d(20),
            nn.Dropout(Dropout),
            nn.Linear(in_features=20, out_features=1))

    def forward(self, x):
        x = self.convolutional_layer(x)
        x = torch.flatten(x, 1)
        x = self.linear_layer(x)
        x = torch.sigmoid(x) 
        return x

#Training and Validatiuon Accuracy Plot
def acc_plot(train, val):
         now = datetime.now()
         dt_string = now.strftime("%d/%m/%Y %H:%M")
         plt.figure(figsize=(16, 5))
         plt.xlabel('EPOCHS')
         plt.ylabel('ACCURACY')

         plt.plot(train, 'r', label='Train')
         plt.plot(val, 'b', label='Val')
         plt.suptitle(("date and time =", dt_string))
         plt.legend()
         plt.show()

#Training and Validatiuon Loss Plot
def loss_plot(train, val):
         now = datetime.now()
         dt_string = now.strftime("%d/%m/%Y %H:%M")
         plt.figure(figsize=(16, 5))
         plt.xlabel('EPOCHS')
         plt.ylabel('LOSS')

         plt.plot(train, 'r', label='Train')
         plt.plot(val, 'b', label='Val')
         plt.suptitle(("date and time =", dt_string))
         plt.legend()
         plt.show()

def Confusion_Matrix(output_array, label_array):
    label_array = np.array([np.array(p.int()) for p in label_array])
    output_array = np.array([np.array(k) for k in output_array])
    my_rounded_list = np.round(output_array, 0) #converting the predicted values from floats to binary (1, 0)
    cm = confusion_matrix(label_array, my_rounded_list)
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

#Training Script
def train(train_loader, criterion, optimizer):
    running_acc = []
    running_loss = []
    for i, (images, target) in enumerate(train_loader):
        #Training mode and sending data to the model
        model.train()
        images = images.to(device)
        target = target.to(device)
        output = model(images)
        
        #Loss
        loss = criterion(output, target)
        loss_temp = loss.item()
        running_loss.append(loss_temp)
        Epoch_loss = np.average(running_loss)
        
        #Converting variables from GPU to CPU (Required for tracking metrics)
        output2 = output.clone().detach()
        output3 = output2.cpu()
        target2 = target.clone().detach()
        target3 = target2.cpu()
        
        #Accuracy
        accuracy = accuracy_score(target3.int(), output3 > 0.5)
        acc = accuracy
        acc = acc * 100
        running_acc.append(acc)
        Epoch_acc = np.average(running_acc)
        
        #Forward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return Epoch_loss, Epoch_acc

#Validation Script
def validate(val_loader, criterion):
    with torch.no_grad():
        running_Vacc = []
        running_Vloss = []
        running_outputs = []
        running_labels = []
        for j, (Vimages, Vtarget) in enumerate(val_loader):
            #Validation mode
            model.eval()
            Vimages = Vimages.to(device)
            Vtarget = Vtarget.to(device)
            Voutput = model(Vimages)
            
            Vloss = criterion(Voutput, Vtarget)
            Vloss = Vloss.item()
            running_Vloss.append(Vloss)
            Epoch_Vloss = np.average(running_Vloss)
            
            Voutput2 = Voutput.clone().detach()
            Voutput3 = Voutput2.cpu()
            Vtarget2 = Vtarget.clone().detach()
            Vtarget3 = Vtarget2.cpu()
            
            Vaccuracy = accuracy_score(Vtarget3.int(), Voutput3 > 0.5)
            Vacc = Vaccuracy
            Vacc = Vacc * 100
            running_Vacc.append(Vacc)
            Epoch_Vacc = np.average(running_Vacc)    
            
            #Tracking Outputs and labels to make a confusion matrix
            running_outputs.extend(Voutput3)
            running_labels.extend(Vtarget3)
    
    return Epoch_Vloss, Epoch_Vacc, running_outputs, running_labels

def Test(test_loader, criterion):
    with torch.no_grad():
        running_Test_acc = []
        running_Test_loss = []
        running_Test_outputs = []
        running_Test_labels = []
        for k, (Test_Images, Test_Targets) in enumerate(test_loader):
            #Test mode
            model.eval()
            Test_Images = Test_Images.to(device)
            Test_Targets = Test_Targets.to(device)
            Test_output = model(Test_Images)
            
            Test_loss = criterion(Test_output, Test_Targets)
            Test_loss = Test_loss.item()
            running_Test_loss.append(Test_loss)
            Epoch_Test_loss = np.average(running_Test_loss)
            
            Test_output2 = Test_output.clone().detach()
            Test_output3 = Test_output2.cpu()
            Test_Targets2 = Test_Targets.clone().detach()
            Test_Targets3 = Test_Targets2.cpu()
            
            Test_accuracy = accuracy_score(Test_Targets3.int(), Test_output3 > 0.5)
            Test_acc = Test_accuracy
            Test_acc = Test_acc * 100
            running_Test_acc.append(Test_acc)
            Epoch_Test_acc = np.average(running_Test_acc)    
            
            #Tracking Outputs and labels to make a confusion matrix
            running_Test_outputs.extend(Test_output3)
            running_Test_labels.extend(Test_Targets3)
    
    return Epoch_Test_loss, Epoch_Test_acc, running_Test_outputs, running_Test_labels

#Epoch iteration function to track training and validation in real time
def train_val_test(train_data, val_data, test_loader, scheduler, criterion, optimizer, patience):
    loss_array = []
    Vloss_array = []
    acc_array = []
    Vacc_array = []

    
    #Tracking start time
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M")
    print("Start:", dt_string)
    
    #from https://github.com/Bjarten/early-stopping-pytorch
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    #Epoch itteration
    for epoch in range(num_epochs):
        print("Epoch", int(epoch + 1/num_epochs))
        print("LR: ", optimizer.state_dict()['param_groups'][0]['lr']) #Prints the LR for each epoch so you can see the change
        #Running the Training and Validation functions
        Epoch_loss, Epoch_acc = train(train_data, criterion, optimizer)
        Epoch_Vloss, Epoch_Vacc, running_outputs, running_labels = validate(val_data, criterion)
        
        #Printing the metrics for each epoch  
        print(f' TLoss: {Epoch_loss:.4f}, TAccuracy: {Epoch_acc:.4f} \n VLoss: {Epoch_Vloss:.4f}, VAccuracy: {Epoch_Vacc:.4f}')
        
        #Tracking metrics
        loss_array.append(Epoch_loss)
        Vloss_array.append(Epoch_Vloss)
        acc_array.append(Epoch_acc)
        Vacc_array.append(Epoch_Vacc)
      
        #scheduler.step(Epoch_Vloss) #plateauLR
        scheduler.step() #stepLR
        
        early_stopping(Epoch_Vloss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
    #end time
    model.load_state_dict(torch.load('checkpoint.pt'))
    Epoch_Test_loss, Epoch_Test_acc, running_Test_outputs, running_Test_labels = Test(test_loader, criterion)
    print(f' \n Test_Loss: {Epoch_Test_loss:.4f}, Test_Accuracy: {Epoch_Test_acc:.4f}')
    
    nownow = datetime.now()
    dt_string2 = nownow.strftime("%d/%m/%Y %H:%M")
    print("Finish Time =", dt_string2)
    
    #Plotting metrics
    print("LR: ", optimizer.state_dict()['param_groups'][0]['lr']) #printing the last LR
    Confusion_Matrix(running_Test_outputs, running_Test_labels)
    acc_plot(acc_array, Vacc_array)
    loss_plot(loss_array, Vloss_array)
    
    #Save model
    model_paths = r"C:\Users\samue\Desktop\PhD\Model Work\Saved_Models\ABIDE_V1.pt"
    torch.save(model, model_paths)
    
    return Epoch_Vacc


#idea from here https://stackoverflow.com/questions/74920920/pytorch-apply-data-augmentation-on-training-data-after-random-split
class TrDataset(Dataset):
  def __init__(self, base_dataset, transformations):
    super(TrDataset, self).__init__()
    self.base = base_dataset
    self.transformations = transformations

  def __len__(self):
    return len(self.base)

  def __getitem__(self, idx):
    x, y = self.base[idx]
    return self.transformations(x), y


def Data_Manager(dataset, Rotate_prob, Flip_prob):
    
    train_transforms = transforms.Compose([RandRotate90(Rotate_prob).set_random_state(seed=42), RandFlip(Flip_prob).set_random_state(seed=42)]) #transforms.Normalize(0.5, 0.25), Resize((61, 61, 61))
    
    
    file_path = r"GIG-ICA_Names_and_Labels.csv"
    label_info = pd.read_csv(file_path, header='infer')
    stratify_labels = np.array(label_info.iloc[:,2])
    Train_temp, Val_temp = train_test_split(dataset, test_size=0.198, random_state=42, stratify=stratify_labels)
    
    
    Train_Sample = TrDataset(Train_temp, train_transforms)
    #Val_aug = TrDataset(Val_temp, validation_transforms)
    
    Val_Sample, Test_Sample  = train_test_split(Val_temp, test_size=0.5, random_state=42)
    
    print(len(Train_Sample))
    print(len(Val_Sample))
    print(len(Test_Sample))
    
    
    trainloader = torch.utils.data.DataLoader(
                  Train_Sample,
                  batch_size=batch_size, pin_memory=True)
    valloader = torch.utils.data.DataLoader(
                  Val_Sample,
                  batch_size=batch_size, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
                  Test_Sample,
                  batch_size=batch_size, pin_memory=True)
    
    return trainloader, valloader, testloader
    

    
#The main section where parameters are defined and functions are run 
if __name__ == '__main__':
    torch.manual_seed(42)
    base_transforms = transforms.Compose([transforms.Normalize(0.5, 0.25), Resize((61, 61, 61))])
    dataset = RandomFmriDataset(transform=base_transforms)
    device = torch.device("cuda")
    
    Dropout = 0.5 
    Rotate_prob = 0.5
    Flip_prob = 0
    
    model = CNN_model(Dropout).cuda()
    num_epochs = 100 
    batch_size = 52  
    learning_rate = 0.0001 # 0.0000008
    criterion = torch.nn.BCELoss(reduction='mean').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=0.90) #not on atm
    validation_split = 0.2
    test_split = 0.5
    shuffle_dataset = True
    random_seed = 42
    patience = 15

    trainloader, valloader, testloader = Data_Manager(dataset, Rotate_prob, Flip_prob)
    
    # train_indices, val_indices, test_indicies = data_indicies(dataset, validation_split, test_split, shuffle_dataset, random_seed)
    # trainloader, valloader, testloader = dataloader_and_transforms(Rotate_prob, Flip_prob, train_indices, val_indices, test_indicies)
    train_val_test(trainloader, valloader, testloader, scheduler, criterion, optimizer, patience)