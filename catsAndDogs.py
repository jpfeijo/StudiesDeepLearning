import tensorflow as tf
import torch
from torch import nn, optim
from torchvision import datasets, transforms
import numpy as np
from PIL import Image

# Criando o modelo

classifier = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
    nn.ReLU(),
    nn.BatchNorm2d(num_features=32),
    nn.MaxPool2d(kernel_size=2),
    nn.Conv2d(32,32,3),
    nn.ReLU(),
    nn.BatchNorm2d(32),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(in_features=14*14*32, out_features=128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128,128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128,1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = optim.Adam(classifier.parameters())

# Base de Dados

data_dir_train = 'dataset/dataset/training_set'
data_dir_test = 'dataset/dataset/test_set'

transform_test = transforms.Compose(
    [
        transforms.Resize([64,64]),
        transforms.ToTensor()
    ]
)

tranform_train = transforms.Compose(
    [
        transforms.Resize([64,64]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=7, translate=(0, 0.07), shear=0.2, scale=(1, 1.2)),
        transforms.ToTensor()
    ]
)

train_dataset = datasets.ImageFolder(data_dir_train, transform=tranform_train)
train_dataset

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = datasets.ImageFolder(data_dir_test, transform=transform_test)
test_dataset

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

# Treinando o Modelo

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device

classifier.to(device)

def training_loop(loader, epoch):
    running_loss = 0.
    running_accuracy = 0.

    for i, data in enumerate(loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = classifier(inputs)

        loss = criterion(outputs, labels.float().view(*outputs.shape))
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predicted = torch.tensor([1 if output > 0.5 else 0 for output in outputs]).to(device)

        equals = predicted == labels.view(*predicted.shape)

        accuracy = torch.mean(equals.float())
        running_accuracy += accuracy

        print('\rÉPOCA {:3d} - Loop {:3d} de {:3d}: perda {:03.2f} - precisão {:03.2f}'.format(epoch + 1, i + 1, len(loader), loss, accuracy), end = '\r')
        
    print('\rÉPOCA {:3d} FINALIZADA: perda {:.5f} - precisão {:.5f}'.format(epoch + 1, running_loss/len(loader), 
                running_accuracy/len(loader)))

#for epoch in range(10):
#    print('Treinando...')
#    training_loop(train_loader, epoch)
#    classifier.eval()
#    print('Validando...')
#    training_loop(test_loader, epoch)
#    classifier.train()


# Avaliação do Modelo

def classify_image(fname):
    import matplotlib.pyplot as plt
    imagem_teste = Image.open(data_dir_test + '/' + fname)
    plt.imshow(imagem_teste)

    imagem_teste = imagem_teste.resize((64, 64))
    imagem_teste = np.array(imagem_teste.getdata()).reshape(*imagem_teste.size, 3)
    imagem_teste = imagem_teste / 255
    imagem_teste = imagem_teste.transpose(2, 0, 1)
    imagem_teste = torch.tensor(imagem_teste, dtype=torch.float).view(-1,*imagem_teste.shape)

    classifier.eval()
    imagem_teste = imagem_teste.to(device)
    output = classifier.forward(imagem_teste)
    if output > 0.5:
        output = 1
    else:
        output = 0
    print('Previsão: ', output)

    idx_to_class = {value: key for key, value in test_dataset.class_to_idx.items()}

    return idx_to_class[output]

imagem = 'gato/cat.3550.jpg'
classify_image(imagem)