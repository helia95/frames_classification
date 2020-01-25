import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from model_def import cnnClassifier
from torch.autograd import Variable
import torchvision.models as models
import sys
import matplotlib.pyplot as plt
import numpy as np
from reader import readShortVideo
import csv
from PIL import Image
import torchvision.transforms as transforms

args = sys.argv

def draw_result(lst,out_dir, F):

    fig = plt.figure(figsize=(10, 10))
    lst_iter = range(len(lst))
    if F:
        l1 = 'Train Loss'
        #l2 = 'Valid loss'
        t = 'Loss'
    else:
        #l1 = 'Train acc'
        l1 = 'Valid acc'
        t = 'Accuracy'


    plt.plot(lst_iter, lst, '-r', label=l1)

    plt.xlabel("n iteration")
    plt.legend(loc='upper left')
    plt.title(t)

    # save image
    plt.savefig('{}.png'.format(out_dir))  # should before show method
    plt.close(fig)

def plot_embedding(X, y, d, title=None, imgName=None):
    # normalization
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    fig = plt.figure(figsize=(10,10))

    groups = np.unique(d)

    for g in groups:
        ii = np.where(y == g)
        plt.scatter(X[ii,0], X[ii,1], label=g)

    plt.xticks([]), plt.yticks([])

    # If title is not given, we assign training_mode to the title.

    plt.title(title)
    plt.legend()

    fig.savefig(imgName)
    plt.close(fig)


def extractor(video_path, file_gt, net, ft_size):

    trans = transforms.Compose([

        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ])

    name = []
    cat = []

    with open(file_gt) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count > 0:

                name.append(row[1])
                cat.append(row[2])

                line_count += 1
            else:
                line_count += 1

    out_labels = []
    out_features = []
    net.cuda()
    for i in range(len(name)):
        video = readShortVideo(video_path, cat[i], name[i])

        tot_frame = []
        for k in range(video.shape[0]):
            tot_frame.append( trans(Image.fromarray(video[k, :, :, :])) )

        tot_frame = torch.stack(tot_frame, 0)

        with torch.no_grad():
            tot_frame = tot_frame.cuda()
            features = net(tot_frame).cpu().view(-1, ft_size)
            out_features.append(features)

    return out_features


path_video = os.path.abspath(args[1])
path_gt = os.path.abspath(args[2])
path_output = os.path.abspath(args[3])

path_model = os.path.join( os.getcwd(), 'model-cnn.pth')

feature_extractor = models.vgg16(pretrained=True).features.cuda()
ft_size = 512*7*7
in_features = extractor(path_video, path_gt,feature_extractor, ft_size=ft_size)


classifier = cnnClassifier()
load_model = torch.load(path_model)

classifier.load_state_dict(load_model)
classifier.cuda()

#iter_tensor = zip(in_features, in_labels)

curr_acc = 0
cl = []
tot_features = []

classifier.eval()
feature_extractor.eval()
tot_features = []
with torch.no_grad():
        for ft in in_features:

            ft = Variable(ft)
            if torch.cuda.is_available():
                ft = ft.cuda()

            tot_features.append( torch.mean(ft,0).cpu().unsqueeze(0))
            output = classifier(torch.mean(ft, 0).unsqueeze(0))

            _, pred = torch.max(output, dim=1)
            cl.append(pred.cpu().numpy())


# Print output
if not os.path.exists(path_output):
    os.makedirs(path_output)

filename = os.path.join(path_output, 'p1_valid.txt')
with open(filename, 'w') as f:
    for item in cl:
        f.write("%s\n" % item[0])