from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch
import numpy as np
from evaluation import Evaluator
from tools.my_dataset import COVIDDataset
import random
import os
from task import DWT_models
from DWT_models import load_DwtSa_weights      # comment this import when training the original models

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed = 42
set_seed(seed)

print("Set your own dirs")
project_dir = "xxx"                                                            # code dir
data_dir = "xxx/covid_5_fold/covid_data1.pkl"                                  # POCUS dataset dir
train_log_txt = project_dir + "train/resnet18_MSHFFA_112_56/log.txt"           # make sure the dir exists
model_save_dir = project_dir + "train/resnet18_MSHFFA_112_56/"
if not os.path.exists(model_save_dir):
    os.mkdir(model_save_dir)
writer = SummaryWriter(project_dir + "train/resnet18_MSHFFA_112_56/logs_train")
f = open(train_log_txt, mode='w', newline='')

# Original models
# M1 = timm.create_model('resnet18', pretrained=True, num_classes=3).cuda()
# M1 = timm.create_model('efficientnet_b0.ra_in1k', pretrained=True, num_classes=3).cuda()
# M1 = timm.create_model('efficientnet_b4.ra2_in1k', pretrained=True, num_classes=3).cuda()
# M1 = DWT_models.vanillaNet_8_Ori(pretrained=True, num_classes=3).cuda()
# M1 = DWT_models.vanillaNet_5_Ori(pretrained=True, num_classes=3).cuda()
# M1 = DWT_models.resnet50_Ori(pretrained=True, num_classes=3).cuda()


M1 = DWT_models.resnet18_DwtSa(pretrained=True, num_classes=3).cuda()
# M1 = DWT_models.efficientNet_b1_DwtSa(pretrained=True, num_classes=3).cuda()
# M1 = DWT_models.efficientNet_b4_DwtSa(pretrained=True, num_classes=3).cuda()
# M1 = DWT_models.resnet50_DwtSa(pretrained=True, num_classes=3).cuda()
# M1 = DWT_models.vanillaNet_5_DwtSa(pretrained=True, num_classes=3).cuda()
# M1 = DWT_models.vanillaNet_8_DwtSa(pretrained=True, num_classes=3).cuda()

# checkpoint = torch.load(project_dir + "train/resnet18_dwtsa/best_model.pth", map_location='cuda')
# M1 = load_DwtSa_weights(checkpoint,M1)

n_parameters = sum(p.numel() for p in M1.parameters() if p.requires_grad)
print('--------------number of params--------------:', n_parameters)

loss_fn = nn.CrossEntropyLoss().cuda()
epoch = 200
weight_decay = 1e-4   # default = 1e-4
learning_rate = 1e-3  # 1e-3
optimizer = torch.optim.Adam(M1.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=0, last_epoch=-1)     # set learning rate decay strategy
evaluator = Evaluator(3)
total_train_step = 0
total_test_step = 0
BS = 128
# best acc
max_acc = 0.
best_info = []


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(0.8, 1.25)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
])

train_data = COVIDDataset(data_dir=data_dir, train=True, transform=train_transform)
test_data = COVIDDataset(data_dir=data_dir, train=False, transform=valid_transform)

# DataLoder
def _init_fn(worker_id):
    np.random.seed(int(seed)+worker_id)
train_loader = DataLoader(dataset=train_data, batch_size=BS, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=_init_fn)  # shuffle: 先打乱所有数据，在按顺序从头到尾取batch
test_loader = DataLoader(dataset=test_data, batch_size=BS, num_workers=4, worker_init_fn=_init_fn)
# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)

print("Training set len：{}".format(train_data_size))
print("Test set len：{}".format(test_data_size))
print(type(train_data))

set_seed(seed)      # needed

beat = []
for i in range(epoch):
    print("------------MSHFFA iter {} starts------------".format(i + 1))
    loss_mean = 0
    iter_in_one_epoch = 0
    M1.train()
    for j, data in enumerate(train_loader):
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = M1(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        iter_in_one_epoch += 1
        loss_mean += loss.item()
        if total_train_step % 100 == 0:
            print("Training iters：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
    loss_this_batch = loss_mean / iter_in_one_epoch
    print('Learning rate this epoch:', scheduler.get_last_lr()[0])
    print("Test Loss：{}".format(loss_this_batch))
    scheduler.step()  # updata learning rate

    # test
    M1.eval()
    evaluator.reset()
    total_test_loss = 0

    with torch.no_grad():
        for j, data in enumerate(test_loader):
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = M1(imgs)
            loss = loss_fn(outputs, targets)
            outputs = torch.softmax(outputs, dim=1)
            pred = outputs.data.cpu().numpy()
            target = targets.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            evaluator.addBatch(pred, target)

            total_test_loss = total_test_loss + loss.item()

    loss_mean_test = total_test_loss / test_loader.__len__()
    print("Test Loss: {}".format(total_test_loss/test_loader.__len__()))

    writer.add_scalar("test_loss", total_test_loss, total_test_step)

    total_test_step = total_test_step + 1

    Acc = evaluator.pixelAccuracy()
    Acc_class = evaluator.classPixelAccuracy()
    Recall_calss = evaluator.classPixelRecall()
    f1_class = evaluator.classF1()
    print("Test mean ACC：{}".format(Acc))
    print("Test ACC：{}".format(Acc_class))
    print("Test Recall: {}".format(Recall_calss))
    print("Test F1-score: {}".format(f1_class))
    print("Test mean F1-score: {}".format(f1_class.mean()))

    f.write(str(i+1)+" "+str(loss_this_batch)+" "+str(loss_mean_test)+" "+str(Acc)+" "+str(Acc_class)+" "+str(Recall_calss)+" "+str(f1_class)+"\n")
    if Acc > max_acc:  # record best accuracy
        max_acc = Acc
        torch.save(M1.state_dict(), model_save_dir + "best_model.pth")  # 保存每一轮训练的结果
        best = [i+1, loss_this_batch, loss_mean_test, Acc, Acc_class, Recall_calss, f1_class]
        print("save the best:epoch:{},train_loss:{},test_loss:{}, test_acc:{}, recall:{}, f1:{}, mean f1:{}".format(i+1, loss_this_batch, loss_mean_test, Acc, Recall_calss, f1_class, f1_class.mean()))
    print("best:", best)
f.close()
writer.close()