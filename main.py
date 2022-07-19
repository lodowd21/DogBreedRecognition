# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFile
import cv2
import glob


# importing Pytorch model libraries
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#importing GUI.py
import gui
print("imported gui.py")

print("hi, imported libaries successfully")

# frog = np.array([1, 2, 3])
# print(frog)

#files = glob.glob("*")
#print(files)

# frog = np.array(glob.glob("./data2/*/*/*/*"))
# print("Frog glob: " , frog)
# print("Frob glob length", len(frog))

dog_files = np.array(glob.glob("./data2/*/*/*/*"))
# print number of images in each dataset
print('There are %d total dog images.' % len(dog_files))
print("Past step 0")
print("")

# Loading images data into the memory and storing them into the variables
data_dir = './data2/dog_images'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
print("Step 1 variables declared")
print("")

# Applying Data Augmentation and Normalization on images
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
print("Applied data augmentation and Normalization on images")

# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=20, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=20, shuffle=False)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=20, shuffle=False)
loaders_transfer = {'train': trainloader,
                    'valid': validloader,
                    'test': testloader}
data_transfer = {
    'train': trainloader
}
print("Past step 1")

# Loading vgg11 into the variable model_transfer
model_transfer = models.vgg11(pretrained=True)
print("pretrained model set")
print("Past step 2")

# Freezing the parameters
for param in model_transfer.features.parameters():
    param.requires_grad = False

# Changing the classifier layer
model_transfer.classifier[6] = nn.Linear(4096, 133, bias=True)

# Moving the model to GPU-RAM space
use_cuda = False
if use_cuda:
    model_transfer = model_transfer.cuda()
print(model_transfer)
print("freezing")

### Loading the Loss-Function
criterion_transfer = nn.CrossEntropyLoss()

### Loading the optimizer
optimizer_transfer = optim.SGD(model_transfer.parameters(), lr=0.001, momentum=0.9)
print("loss function")
print("passed step 3")


def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.inf  # --->   Max Value (As the loss decreases and becomes less than this value it gets saved)
    count = 0
    print("about to train")

    for epoch in range(1, n_epochs + 1):
        # Initializing training variables
        train_loss = 0.0
        valid_loss = 0.0
        # Start training the model
        print("Starting training for epoch: ", epoch)
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU's memory space (if available)
            if use_cuda:
                data, target = data.cuda(), target.cuda()
                model.to('cuda')
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            count += 1
            print(count, " train loss: ", train_loss)

        # validate the model #
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            accuracy = 0
            # move to GPU's memory space (if available)
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## Update the validation loss
            logps = model(data)
            loss = criterion(logps, target)

            valid_loss += ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
            print("valid loss: ", valid_loss)

        # print both training and validation losses
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
        ))

        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            # Saving the model
            torch.save(model.state_dict(), 'model_transfer.pt')
            valid_loss_min = valid_loss
    # return the trained model
    return model


print("passed step 4")

# train the model
model_transfer = train(1, loaders_transfer, model_transfer, optimizer_transfer, criterion_transfer, use_cuda,
                       'model_transfer.pt')

# load the model that got the best validation accuracy
model_transfer.load_state_dict(torch.load('model_transfer.pt'))
print("passed step 5")


def test(loaders, model, criterion, use_cuda):
    # Initializing the variables
    test_loss = 0.
    correct = 0.
    total = 0.
    print("in test function")

    model.eval()  # So that it doesn't change the model parameters during testing
    for batch_idx, (data, target) in enumerate(loaders['test']):
        print("batch_idx", batch_idx)
        # move to GPU's memory spave if available
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # Passing the data to the model (Forward Pass)
        output = model(data)
        loss = criterion(output, target)  # Test Loss
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # Output probabilities to the predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # Comparing the predicted class to output
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)

    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))


def predict_breed_transfer(img_path):
    print("in predict_breed_transfer 1")

    # Preprocessing the input image
    transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    print("in predict_breed_transfer 2")
    print(img_path)

    # img = Image.open(r"C:\Users\liamo\Soft_Engr\my_test_images\Airedale_terrier_00154.jpg")
    img = Image.open(img_path, 'r')  # This method will show image in any image viewer

    # img.show()

    print("testing image after displayed on screen")

    # img = Image.open(img_path)
    img = transform(img)[:3, :, :].unsqueeze(0)
    print("in predict_breed_transfer 3, image opened")

    if use_cuda:
        img = img.cuda()
        model_transfer.to('cuda')
        # Passing throught the model
    print("in predict_breed_transfer 4")

    model_transfer.eval()
    print("in predict_breed_transfer 5")

    # Checking the name of class by passing the index
    class_names = [item[4:].replace("_", " ") for item in data_transfer['train'].dataset.classes]

    idx = torch.argmax(model_transfer(img))
    return class_names[idx]

    output = model_transfer(img)
    # Probabilities to class
    pred = output.data.max(1, keepdim=True)[1]
    return pred


print("starting step 6")
test(loaders_transfer, model_transfer, criterion_transfer, use_cuda)
print("passed step 6")

print("Before step 7")
# Loading the new image directory
dog_files_short = np.array(glob.glob("my_test_images/*"))
print(dog_files_short)
print('There are %d total files in dog_files_short.' % len(dog_files_short))

# Loading the model
model_transfer.load_state_dict(torch.load('model_transfer.pt'))

print("step 8, trying to predict dog breed 122")
# dog_files_predict1 = np.array(glob.glob("./my_test_images/Airedale_terrier_00154.jpg"))
# dog_files_predict2 = np.array(glob.glob("./my_test_images/Akita_00270.jpg"))
# dog_files_predict3 = np.array(glob.glob("./my_test_images/Golden_retriever_05220.jpg"))

prediction1 = predict_breed_transfer("C:/Users/liamo/Soft_Engr/my_test_images/Airedale_terrier_00154.jpg")
prediction2 = predict_breed_transfer("C:/Users/liamo/Soft_Engr/my_test_images/Akita_00270.jpg")
prediction3 = predict_breed_transfer("C:/Users/liamo/Soft_Engr/my_test_images/Golden_retriever_05220.jpg")
prediction4 = predict_breed_transfer("C:/Users/liamo/Soft_Engr/my_test_images/Troy_haven_logo.jpg")
prediction5 = predict_breed_transfer("C:/Users/liamo/Soft_Engr/my_test_images/akita2.jpg")
prediction6 = predict_breed_transfer("C:/Users/liamo/Soft_Engr/my_test_images/golden.jpg")
prediction7 = predict_breed_transfer("C:/Users/liamo/Soft_Engr/my_test_images/beagle.jpg")

prediction8 = predict_breed_transfer("C:/Users/liamo/Soft_Engr/my_test_images/pointer.jpg")
prediction9 = predict_breed_transfer("C:/Users/liamo/Soft_Engr/my_test_images/bernard.jpg")
prediction10 = predict_breed_transfer("C:/Users/liamo/Soft_Engr/my_test_images/kom.jpg")
prediction11 = predict_breed_transfer("C:/Users/liamo/Soft_Engr/my_test_images/dane.jpg")
prediction12 = predict_breed_transfer("C:/Users/liamo/Soft_Engr/my_test_images/dachshund.jpg")
prediction13 = predict_breed_transfer("C:/Users/liamo/Soft_Engr/my_test_images/boston.jpeg")
prediction14 = predict_breed_transfer("C:/Users/liamo/Soft_Engr/my_test_images/pointer2.jpg")
prediction15 = predict_breed_transfer("C:/Users/liamo/Soft_Engr/my_test_images/bernard2.jpg")

prediction16 = predict_breed_transfer("C:/Users/liamo/Soft_Engr/my_test_images/kom2.jpg")
prediction17 = predict_breed_transfer("C:/Users/liamo/Soft_Engr/my_test_images/dane2.jpg")
prediction18 = predict_breed_transfer("C:/Users/liamo/Soft_Engr/my_test_images/beagle2.jpg")
prediction19 = predict_breed_transfer("C:/Users/liamo/Soft_Engr/my_test_images/boston2.jpg")
prediction20 = predict_breed_transfer("C:/Users/liamo/Soft_Engr/my_test_images/dachshund2.jpg")

print("dog breed Airedale prediction: ", prediction1)
print("dog breed Airedale prediction 2: ", prediction4)
print("")
print("dog breed Akita prediction: ", prediction2)
print("dog breed Akita prediction 2: ", prediction5)
print("")

print("dog breed Golden Retriever prediction: ", prediction3)
print("dog breed Golden Retriever prediction 2: ", prediction6)
print("")

print("dog breed Beagle prediction: ", prediction7)
print("dog breed Beagle prediction 2: ", prediction18)
print("")

print("dog breed Pointer prediction: ", prediction8)
print("dog breed Pointer prediction 2: ", prediction14)
print("")

print("dog breed Saint-Bernard prediction: ", prediction9)
print("dog breed Saint-Bernard prediction 2: ", prediction15)
print("")

print("dog breed Komondor prediction: ", prediction10)
print("dog breed Komondor prediction 2: ", prediction16)
print("")

print("dog breed Great Dane prediction: ", prediction11)
print("dog breed Great Dane prediction 2: ", prediction17)
print("")

print("dog breed Dachshund prediction: ", prediction12)
print("dog breed Dachshund prediction 2: ", prediction20)
print("")

print("dog breed Boston Terrier prediction: ", prediction13)
print("dog breed Boston Terrier prediction 2: ", prediction19)

print("all done!")

