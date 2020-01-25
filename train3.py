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
path_dataset = '../features_FullVideo/'
path_checkpoint = './models/'
path_data = '../res3/'
load_path = './models/model-rnn.pth'

batch_sz = 32

# Dataset
train_cut_features = torch.load('{}train_cut_features.pt'.format(path_dataset))
train_cut_labels = torch.load('{}train_cur_label.pt'.format(path_dataset))
train_cut_len = torch.load('{}train_cur_len.pt'.format(path_dataset))

valid_cut_features = torch.load('{}valid_cut_features.pt'.format(path_dataset))
vald_cut_labels = torch.load('{}valid_cur_label.pt'.format(path_dataset))
valid_cut_len = torch.load('{}valid_cur_len.pt'.format(path_dataset))

train_iterator = list(zip(train_cut_features, train_cut_labels, train_cut_len))
valid_iterator = list(zip(valid_cut_features, vald_cut_labels, valid_cut_len))

# Training Parameters

epochs = 30

# Net
net = model_def.seq_to_seq()
#net.apply(model_def.weight_init)
net.load_state_dict(torch.load(load_path))

if torch.cuda.is_available():
    net.cuda()

# Optimizer
optimizer = Adam(net.parameters(), lr=0.0002)

# Loss function
criterion = nn.CrossEntropyLoss()

# Start training process
train_loss = []
valid_loss = []
train_acc = []
valid_acc = []


for epoch in range(epochs):

    random.shuffle(train_iterator)
    batches = list(chunks(train_iterator, batch_sz))

    # Training
    net.train()
    t_loss = 0.0
    t_acc = 0

    for i in range(len(batches)):
        ft, lb, leng = zip(*batches[i])
        cur_batch = len(lb)

        loss = 0
        for j in range(cur_batch):
            batch = ft[j]
            target = lb[j]
            ll = leng[j]
            batch, target = Variable(batch), Variable(target)
            batch = batch.cuda()
            target = target.cuda()

            output = net(batch)

            p_loss = criterion(output, target)
            loss += p_loss / cur_batch

            _, pred = torch.max(output, dim=1)
            correct = pred.eq(target.data.view_as(pred))
            accuracy = torch.mean(correct.type(torch.FloatTensor))
            t_acc += accuracy.item() / cur_batch

        loss.backward()
        optimizer.step()


        t_loss += loss.item()

    # Validation
    net.eval()
    batches = list(chunks(valid_iterator, batch_sz))

    v_loss = 0.0
    v_acc = 0
    with torch.no_grad():
        for i in range(len(batches)):
            ft, lb, leng = zip(*batches[i])
            cur_batch = len(lb)

            loss = 0
            for j in range(cur_batch):
                batch = ft[j]
                target = lb[j]
                ll = leng[j]

                batch, target = Variable(batch), Variable(target)
                batch = batch.cuda()
                target = target.cuda()

                output = net(batch)

                p_loss = criterion(output, target)
                loss += p_loss / cur_batch

                _, pred = torch.max(output, dim=1)
                correct = pred.eq(target.data.view_as(pred))
                accuracy = torch.mean(correct.type(torch.FloatTensor))
                v_acc += accuracy.item() / cur_batch

            v_loss += loss.item()

    print('*************************************************')

    #tot_t_loss = t_loss
    #tot_t_acc = t_acc / len(train_tensor)
    #tot_v_loss = v_loss / len(valid_tensor)
    #tot_v_acc = v_acc / len(valid_tensor)
    print('Epoch [%d/%d], Training loss: %.4f'
          % (epoch + 1, epochs, t_loss))
    print('Epoch [%d/%d], Validation loss: %.4f, Accuracy: %.4f'
          % (epoch + 1, epochs, v_loss, v_acc))

    train_loss.append(t_loss)
    valid_loss.append(v_loss)

    #train_acc.append(tot_t_acc)
    valid_acc.append(v_acc)

    if epoch == epochs-1:
        torch.save(net.state_dict(), '{}model-s2s.pth'.format(path_checkpoint))


pkl.dump(train_loss, open('{}t_loss.p'.format(path_data), 'wb'))
pkl.dump(valid_loss, open('{}v_loss.p'.format(path_data), 'wb'))
#pkl.dump(train_acc, open('{}t_acc.p'.format(path_data), 'wb'))
pkl.dump(valid_acc, open('{}v_acc.p'.format(path_data), 'wb'))
