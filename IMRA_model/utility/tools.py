import torch
import torchvision
import numpy as np
import matplotlib.pylab as plt


def calculate_accuracy_binary(outputs, targets):
    outputs = outputs.data.cpu().numpy().flatten()
    targets = targets.data.cpu().numpy().flatten()
    hit = ((outputs > 0.5) == targets).sum()
    #hit = sum(abs(outputs-targets))
    tsum = targets.shape[0]
    return (hit + 1e-8) / (tsum + 1e-8)


def calculate_accuracy(outputs, targets):
    #outputs = outputs.data.cpu().numpy().flatten()
    #targets = targets.data.cpu().numpy().flatten()
    max_vals, max_indices = torch.max(outputs, 1)
    acc = (max_indices == targets.long()).sum().data.cpu().numpy() / max_indices.size()[0]
    return acc

def image_cat(inputs,bs):
    data=[]
    for h in range(bs):
        data.append(inputs[h, :, :, :])
    data = [x for x in data]
    data_all = torchvision.utils.make_grid(data, nrow=int(np.ceil(np.sqrt(len(data)))), padding=10, normalize=True,
                                           range=None, scale_each=True)

    return data_all


def add_image_unet(inputs,masks,est_maps,outputs, targets, writer, subset, epoch):

    outputs = outputs.data.cpu().numpy()
    targets = targets.data.cpu().numpy()

    # print('image added... with len of {}'.format(len(targets)))

    data_all = image_cat(inputs,targets.shape[0])
    mask_all = image_cat(masks, targets.shape[0])
    estmaps_all = image_cat(est_maps, targets.shape[0])


    if subset == 'val':
        writer.add_image(subset + '_step_' + str(epoch) +  '/diff_'+str(sum(abs(outputs-targets))) + '/gt:' + str(targets) + '/pred:' + str(outputs),
                     img_tensor=data_all, global_step=epoch, dataformats='CHW')
        writer.add_image(subset + '_step_' + str(epoch) + '/diff_' + str(sum(abs(outputs - targets))) + '/gt:' + str(
            targets) + '/pred:' + str(outputs),
                         img_tensor=mask_all, global_step=epoch, dataformats='CHW')
        writer.add_image(subset + '_step_' + str(epoch) + '/diff_' + str(sum(abs(outputs - targets))) + '/gt:' + str(
            targets) + '/pred:' + str(outputs),
                         img_tensor=mask_all, global_step=epoch, dataformats='CHW')
    else:
        writer.add_image(subset + '_step_' + str(epoch ),img_tensor=data_all, global_step=epoch, dataformats='CHW')

def add_image_3d(inputs, outputs, targets, writer, subset, epoch,name):

    outputs = outputs.data.cpu().numpy()
    targets = targets.data.cpu().numpy()

    # print('image added... with len of {}'.format(len(targets)))
    data = []
    for h in range(targets.shape[0]):
        data.append(inputs[h, :,  :, :])
    data = [x for x in data]
    # data = torch.cat(data, dim=0)
    data_all = torchvision.utils.make_grid(data, nrow=int(np.ceil(np.sqrt(len(data)))), padding=10, normalize=True, range=None, scale_each=True)
    # if subset == 'val':
    #     writer.add_image(subset + '_step_' + str(epoch) + '/Diff_'+str(sum(sum(abs(outputs-targets)))) + '/diff_'+str(sum(abs(outputs-targets))) + '/gt:' + str(targets) + '/pred:' + str(outputs),
    #                  img_tensor=data_all, global_step=epoch, dataformats='CHW')
    if subset == 'val':
        # print('val image added')
        writer.add_image(subset + '_step_' + str(epoch) +'/'+ name + '/diff_'+str(sum(abs(outputs-targets))) + '/gt:' + str(targets) + '/pred:' + str(outputs),
                     img_tensor=data_all, global_step=epoch, dataformats='CHW')
    else:
        # print('train image added')
        writer.add_image(subset + '_step_' + str(epoch )+'/'+name,img_tensor=data_all, global_step=epoch, dataformats='CHW')

def add_image(inputs, outputs, targets, writer, subset, epoch):
    outputs = outputs.data.cpu().numpy().flatten()
    targets = targets.data.cpu().numpy().flatten()
    for h in range(targets.shape[0]):
        data = []
        data.append(inputs[h, :, :, :])
        data = [x for x in data]
        data = torch.cat(data, dim=1)
        data_all = torchvision.utils.make_grid(data, nrow=1, padding=2, normalize=False, range=None, scale_each=False)
        writer.add_image(subset + '_step_' + str(epoch) + '/gt: ' + str(targets[h]) + '/pred: ' + str(outputs[h]),
                         img_tensor=data_all, global_step=epoch, dataformats='CHW')

def add_gl_image(images,patches, outputs, targets, writer, subset, epoch):
    outputs = outputs.data.cpu().numpy().flatten()
    targets = targets.data.cpu().numpy().flatten()
    for h in range(targets.shape[0]):
        data_g = []
        data_l = []
        data_g.append(images[h, :, :, :])
        data_l.append(patches[h, :, :, :])
        data_g = [x for x in data_g]
        data_l = [x for x in data_l]
        data_g = torch.cat(data_g, dim=1)
        data_l = torch.cat(data_l, dim=1)
        data_g_all = torchvision.utils.make_grid(data_g, nrow=1, padding=2, normalize=False, range=None, scale_each=False)
        data_l_all = torchvision.utils.make_grid(data_l, nrow=1, padding=2, normalize=False, range=None,
                                                 scale_each=False)
        writer.add_image(subset + '_step_' + str(epoch) + '/diff_'+str(abs(outputs[h]-targets[h])) + '_g_/gt: ' + str(targets[h]) + '/pred: ' + str(outputs[h]),img_tensor=data_g_all, global_step=epoch, dataformats='CHW')
        writer.add_image(subset + '_step_' + str(epoch) + '/diff_'+str(abs(outputs[h]-targets[h])) + '_l_/gt: ' + str(targets[h]) + '/pred: ' + str(outputs[h]),img_tensor=data_l_all, global_step=epoch, dataformats='CHW')

def add_gld_image(images,patches,details, outputs, targets, writer, subset, epoch):
    outputs = outputs.data.cpu().numpy().flatten()
    targets = targets.data.cpu().numpy().flatten()
    for h in range(targets.shape[0]):
        data_g = []
        data_l = []
        data_d = []
        data_g.append(images[h, :, :, :])
        data_l.append(patches[h, :, :, :])
        data_d.append(details[h, :, :, :])
        data_g = [x for x in data_g]
        data_l = [x for x in data_l]
        data_d = [x for x in data_d]
        data_g = torch.cat(data_g, dim=1)
        data_l = torch.cat(data_l, dim=1)
        data_d = torch.cat(data_d, dim=1)
        data_g_all = torchvision.utils.make_grid(data_g, nrow=1, padding=2, normalize=False, range=None, scale_each=False)
        data_l_all = torchvision.utils.make_grid(data_l, nrow=1, padding=2, normalize=False, range=None,
                                                 scale_each=False)
        data_d_all = torchvision.utils.make_grid(data_d, nrow=1, padding=2, normalize=False, range=None,
                                                 scale_each=False)
        writer.add_image(subset + '_step_' + str(epoch) + '/diff_'+str(abs(outputs[h]-targets[h])) + '_g_/gt: ' + str(targets[h]) + '/pred: ' + str(outputs[h]),img_tensor=data_g_all, global_step=epoch, dataformats='CHW')
        writer.add_image(subset + '_step_' + str(epoch) + '/diff_'+str(abs(outputs[h]-targets[h])) + '_l_/gt: ' + str(targets[h]) + '/pred: ' + str(outputs[h]),img_tensor=data_l_all, global_step=epoch, dataformats='CHW')
        writer.add_image(
            subset + '_step_' + str(epoch) + '/diff_' + str(abs(outputs[h] - targets[h])) + '_d_/gt: ' + str(
                targets[h]) + '/pred: ' + str(outputs[h]), img_tensor=data_d_all, global_step=epoch, dataformats='CHW')

def add_gl_image_index(images, patches, outputs, targets, writer, subset, epoch,index):
    outputs = outputs.data.cpu().numpy().flatten()
    targets = targets.data.cpu().numpy().flatten()
    for h in range(targets.shape[0]):
        data_g = []
        data_l = []
        data_g.append(images[h, :, :, :])
        data_l.append(patches[h, :, :, :])
        data_g = [x for x in data_g]
        data_l = [x for x in data_l]
        data_g = torch.cat(data_g, dim=1)
        data_l = torch.cat(data_l, dim=1)
        data_g_all = torchvision.utils.make_grid(data_g, nrow=1, padding=2, normalize=False, range=None,
                                                 scale_each=False)
        data_l_all = torchvision.utils.make_grid(data_l, nrow=1, padding=2, normalize=False, range=None,
                                                 scale_each=False)
        writer.add_image(
            subset + '_step_' + str(epoch)+ '_diff_' + str(outputs[h] - targets[h]) + '_index_' + str(index) + '/g_gt: ' + str(
                targets[h]) + '/pred: ' + str(outputs[h]), img_tensor=data_g_all, global_step=epoch,
            dataformats='CHW')
        writer.add_image(
            subset + '_step_' + str(epoch)+ '_diff_' + str(outputs[h] - targets[h]) + '_index_' + str(index) + '/l_gt: ' + str(
                targets[h]) + '/pred: ' + str(outputs[h]), img_tensor=data_l_all, global_step=epoch,
            dataformats='CHW')

def add_gld_image_index(images, patches, details, outputs, targets, writer, subset, epoch,index):
    outputs = outputs.data.cpu().numpy().flatten()
    targets = targets.data.cpu().numpy().flatten()
    for h in range(targets.shape[0]):
        data_g = []
        data_l = []
        data_d = []
        data_g.append(images[h, :, :, :])
        data_l.append(patches[h, :, :, :])
        data_d.append(details[h, :, :, :])
        data_g = [x for x in data_g]
        data_l = [x for x in data_l]
        data_d = [x for x in data_d]
        data_g = torch.cat(data_g, dim=1)
        data_l = torch.cat(data_l, dim=1)
        data_d = torch.cat(data_d, dim=1)
        data_g_all = torchvision.utils.make_grid(data_g, nrow=1, padding=2, normalize=False, range=None,
                                                 scale_each=False)
        data_l_all = torchvision.utils.make_grid(data_l, nrow=1, padding=2, normalize=False, range=None,
                                                 scale_each=False)
        data_d_all = torchvision.utils.make_grid(data_d, nrow=1, padding=2, normalize=False, range=None,
                                                 scale_each=False)
        writer.add_image(
            subset + '_step_' + str(epoch)+ '_diff_' + str(outputs[h] - targets[h]) + '_index_' + str(index) + '/g_gt: ' + str(
                targets[h]) + '/pred: ' + str(outputs[h]), img_tensor=data_g_all, global_step=epoch,
            dataformats='CHW')
        writer.add_image(
            subset + '_step_' + str(epoch)+ '_diff_' + str(outputs[h] - targets[h]) + '_index_' + str(index) + '/l_gt: ' + str(
                targets[h]) + '/pred: ' + str(outputs[h]), img_tensor=data_l_all, global_step=epoch,
            dataformats='CHW')
        writer.add_image(
            subset + '_step_' + str(epoch) + '_diff_' + str(outputs[h] - targets[h]) + '_index_' + str(
                index) + '/d_gt: ' + str(
                targets[h]) + '/pred: ' + str(outputs[h]), img_tensor=data_d_all, global_step=epoch,
            dataformats='CHW')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

