import torch
import os
from torch import nn
from torch import optim
from time import gmtime, strftime
from tensorboardX import SummaryWriter
import tqdm as tqdm

# os.sys.path.append('/data1/MedicalImage/User/xing/SigmaPy')
from IMRA_model.model.resnet2d_cbam import *
from IMRA_model.dataset.data_generator import *
from IMRA_model.utility.tools import *
from IMRA_model.utility.lr_cosine import CosineAnnealingWarmUpRestarts
from IMRA_model.utility.sampler import BalancedBatchSampler
import warnings
warnings.filterwarnings(action='ignore')



if __name__ == '__main__':
    time_string = strftime("%a%d%b%Y-%H%M%S", gmtime())
    # result_path = r'E:\Data\Medlink\QualityControl\model_train'
    result_path = r'E:\Data\Rimag\train_log'
    descrip = 'resnet34_cbam'
    model_save_path = os.path.join(result_path, descrip, time_string, 'save')
    tb_save_path = os.path.join(result_path, descrip,time_string, 'tb')
    os.makedirs(model_save_path)
    os.makedirs(tb_save_path)
    writer = SummaryWriter(logdir=tb_save_path)

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    # torch.cuda.set_device(1)

    # model = get_unetx_reduce_fpn_v1().cuda()
    # model = resnet.resnet10(in_channels=8, drop_rate=0.3, sample_size=64, sample_duration=16, shortcut_type='B', num_classes=1).cuda()
    # model = resnet_v2.resnet10(in_channels=8, drop_rate=0.3, sample_size=64, sample_duration=16, shortcut_type='B',
    #                         num_classes=1).cuda()
    num_classes = 8
    model = ResNet(dataset='calc', depth=34, num_classes=num_classes).cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-8, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, threshold=0.01, factor=0.3)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=10, T_mult=2, eta_max=1e-3, T_up=2, gamma=0.6)

    trainconfig = {"dataset": 'mammo_calc', "subset": '0'}
    train_config = dataconfig(**trainconfig)
    training_data = DataGenerator(train_config, transform=transforms.ToTensor())
    # train_loader = DataLoader(training_data, num_workers=4, batch_size=16, shuffle= True)
    train_loader = DataLoader(training_data, num_workers=4,
                              sampler=BalancedBatchSampler(training_data, type='single_label'), batch_size=48,drop_last=True)

    valconfig = {"dataset": "calc", "subset": '1'}
    val_config = dataconfig(**valconfig)
    validation_data = DataGenerator(val_config, transform=transforms.ToTensor())
    val_loader = DataLoader(validation_data, num_workers=4,shuffle=True)

    print('data loader finished')

    Train_C_flag = False
    epoch_len = 60

    bst_acc = 0
    bst_loss = 5
    bst_tsh = 0.1

    if Train_C_flag == True:
        model_load_path = r'E:\Xing\mass0508\Train_log\June17_attenres_224\Wed17Jun2020-075951\save'
        model_name = r'\best_model.pth'

        # model_load_path = r'pretrain'
        # model_name = r'\resnet_34.pth'

        checkpoint = torch.load(model_load_path + model_name)
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        Epoch = checkpoint['epoch']
    else:
        Epoch = 0

    for epoch in range(Epoch, Epoch + epoch_len):
        model.train()
        losses = AverageMeter()
        accuracies = AverageMeter()

        for i, (images, labels) in enumerate(train_loader):
            # print(i,images.shape)
            targets = labels.cuda().float()
            outputs = model(images.cuda())
            # print('outputs: ', outputs.data.cpu().numpy().tolist(), 'targets: ', targets.data.cpu().numpy().tolist())
            loss = criterion(outputs, targets.long())
            acc = calculate_accuracy(outputs, targets)
            losses.update(loss.item(), targets.size(0))
            accuracies.update(acc, targets.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch) % 10 == 0 and i % 10 == 0:
                _, predict = torch.max(outputs, 1)
                add_image_3d(images, predict, targets, writer, subset='train', epoch=epoch, name=str(i) + '_image')
            print(i,loss,images.shape)

        losses_val = AverageMeter()
        accuracies_val = AverageMeter()
        model.eval()
        with torch.no_grad():
            for j, (inputs_val, targets_val) in enumerate(val_loader):
                targets_val = targets_val.cuda().float()
                outputs_val = model(inputs_val.cuda())
                loss_val = criterion(outputs_val, targets_val.long())
                acc_val = calculate_accuracy(outputs_val, targets_val)
                losses_val.update(loss_val.item(), targets_val.size(0))
                accuracies_val.update(acc_val, targets_val.size(0))

                if (epoch) % 10 == 0 and j % 50 == 0:
                    print(j, loss_val)
                    _, predict = torch.max(outputs_val, 1)
                    add_image_3d(inputs_val, predict, targets_val, writer, subset='val', epoch=epoch,
                                 name=str(j) + '_image')

        # scheduler.step(losses_val.avg)
        scheduler.step()

        print('epoch: ', epoch + 1, 'train_loss: ', losses.avg, 'train_acc: ', accuracies.avg,
              'val_loss: ', losses_val.avg, 'val_acc: ', accuracies_val.avg)

        if bst_loss >= losses_val.avg or abs(bst_loss - losses_val.avg) <= bst_tsh:

            if bst_acc <= accuracies_val.avg:
                save_file_path = os.path.join(model_save_path, 'best_model.pth')
                states = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(states, save_file_path)
                better_epoch = epoch

                bst_acc = accuracies_val.avg
                bst_loss = losses_val.avg

        print('better model found at epoch {} with val_loss {} and val_acc {}'.format(better_epoch, bst_loss, bst_acc))

        # Save model and print something in the tensorboard
        # Save model and print something in the tensorboard
        writer.add_scalars('loss/epoch',
                           {'train loss': losses.avg, 'validation loss': losses_val.avg}, epoch + 1)
        writer.add_scalars('acc/epoch',
                           {'train accuracy': accuracies.avg, 'validation accuracy': accuracies_val.avg}, epoch + 1)
        writer.add_scalars('Learning Rate/epoch',
                           {'train accuracy': optimizer.param_groups[0]['lr']}, epoch + 1)


