import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch.nn as nn
from torch.optim import Adam
import torch
from torch.autograd import Variable
import model_def
import pickle as pkl
import random

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

# Path
path_dataset = '../features/'
path_checkpoint = './models/'
path_data = '../res1/'
batch_sz = 16

# Dataset
train_tensor = torch.load('{}train_features.pt'.format(path_dataset))
valid_tensor = torch.load('{}valid_features.pt'.format(path_dataset))

# Training Parameters
epochs = 30

# Net
net = model_def.cnnClassifier()
net.apply(model_def.weight_init)

if torch.cuda.is_available():
    net.cuda()

# Optimizer
optimizer = Adam(net.parameters(), lr=0.0001)

# Loss function
criterion = nn.CrossEntropyLoss()

# Start training process
train_loss = []
valid_loss = []
train_acc = []
valid_acc = []


for epoch in range(epochs):

    random.shuffle(train_tensor)

    batches = list(chunks(train_tensor, batch_sz))
    # Training
    net.train()
    t_loss = 0.0
    t_acc = 0

    for i in range(len(batches)):

        ft, lb = zip(*batches[i])

        cur_batch = len(lb)

        batch = []
        for j in range(cur_batch):
            batch.append( torch.Tensor( ft[j].mean(0) ) )

        batch = torch.stack(batch, 0)
        target = torch.LongTensor(lb)

        batch, target = Variable(batch), Variable(target)
        batch = batch.cuda()
        target = target.cuda()

        optimizer.zero_grad()

        output = net(batch)
        loss = criterion(output, target)

        loss.backward()

        optimizer.step()

        t_loss += loss.item()*cur_batch
        _, pred = torch.max(output, dim=1)
        correct = pred.eq(target.data.view_as(pred))
        accuracy = torch.mean(correct.type(torch.FloatTensor))
        t_acc += accuracy.item()*cur_batch

    # Validation
    net.eval()
    batches = list(chunks(valid_tensor, batch_sz))

    v_loss = 0.0
    v_acc = 0
    with torch.no_grad():
        for i in range(len(batches)):

            ft, lb = zip(*batches[i])
            cur_batch = len(lb)

            batch = []
            for j in range(cur_batch):
                batch.append(torch.Tensor(ft[j].mean(0)))

            batch = torch.stack(batch, 0)
            target = torch.LongTensor(lb)

            batch, target = Variable(batch), Variable(target)
            batch = batch.cuda()
            target = target.cuda()

            output = net(batch)
            loss = criterion(output, target)


            v_loss += loss.item() * cur_batch
            _, pred = torch.max(output, dim=1)
            correct = pred.eq(target.data.view_as(pred))
            accuracy = torch.mean(correct.type(torch.FloatTensor))
            v_acc += accuracy.item() * cur_batch

    print('*************************************************')

    tot_t_loss = t_loss / len(train_tensor)
    tot_t_acc = t_acc / len(train_tensor)
    tot_v_loss = v_loss / len(valid_tensor)
    tot_v_acc = v_acc / len(valid_tensor)
    print('Epoch [%d/%d], Training loss: %.4f, Accuracy: %.4f'
          % (epoch + 1, epochs, tot_t_loss, tot_t_acc))
    print('Epoch [%d/%d], Validation loss: %.4f, Accuracy: %.4f'
          % (epoch + 1, epochs, tot_v_loss, tot_v_acc))

    train_loss.append(tot_t_loss)
    valid_loss.append(tot_v_loss)

    train_acc.append(tot_t_acc)
    valid_acc.append(tot_v_acc)

    if epoch == epochs-1:
        torch.save(net.state_dict(), '{}model-cnn.pth'.format(path_checkpoint))


pkl.dump(train_loss, open('{}t_loss.p'.format(path_data), 'wb'))
pkl.dump(valid_loss, open('{}v_loss.p'.format(path_data), 'wb'))
pkl.dump(train_acc, open('{}t_acc.p'.format(path_data), 'wb'))
pkl.dump(valid_acc, open('{}v_acc.p'.format(path_data), 'wb'))
