import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from model_def import seq_to_seq
import torchvision.models as models
import sys
import matplotlib.pyplot as plt
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

def extractor(images_path, net):

    trans = transforms.Compose([

        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ])

    ft_size = 7*7*512
    dir_video = sorted( os.listdir(images_path) )

    net.eval()
    net.cuda()

    all_out_frame = []
    for cat in dir_video:
        path_file = os.path.join(images_path, cat)
        files_img = sorted(os.listdir( path_file ) )

        cat_frames = []
        for fname in files_img:
            img = Image.open( os.path.join( path_file , fname ) )
            img = trans(img)

            with torch.no_grad():
                img = img.cuda()
                features = net(img.unsqueeze(0)).view(-1, ft_size)
                cat_frames.append(features.detach().cpu())

        all_out_frame.append(torch.stack(cat_frames))


    return all_out_frame


path_video = os.path.abspath(args[1])
path_output = os.path.abspath(args[2])

feature_extractor = models.vgg16(pretrained=True).features.cuda()
features = extractor(path_video, feature_extractor)


path_model = os.path.join(os.getcwd(), 'model-s2s.pth')
classifier = seq_to_seq()
load_model = torch.load(path_model)

classifier.load_state_dict(load_model)
classifier.cuda()

curr_acc = 0
cl = []
classifier.eval()
with torch.no_grad():
    for ft in features:
        ft = ft.cuda()
        output = classifier(ft)
        _, pred = torch.max(output, dim=1)
        cl.append(pred.cpu().numpy())


# Output
cat = sorted(os.listdir(path_video))

for i in range(len(cat)):
    with open(os.path.join(path_output, cat[i] + '.txt'), "w") as f:
        for j, pred in enumerate(cl[i]):
            f.write(str(pred))
            if j != len(cl[i]) - 1:
                f.write("\n")