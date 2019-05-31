from torch_lib.Dataset import *
from torch_lib.Model import Model, OrientationLoss


import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg
from torch.utils import data


import os

def main():

    # hyper parameters
    epochs = 100
    batch_size = 256
    alpha = 0.6
    w = 0.4

    print("Loading all detected objects in dataset...")

    train_path = os.path.abspath(os.path.dirname(__file__)) + '/Kitti/training'
    dataset = Dataset(train_path)

    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 16}

    generator = data.DataLoader(dataset, **params)

    my_vgg = vgg.vgg19_bn(pretrained=True)
    print(my_vgg.features)
    model = Model(features=my_vgg.features).cuda()
    opt_SGD = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    conf_loss_func = nn.CrossEntropyLoss().cuda()
    dim_loss_func = nn.MSELoss().cuda()
    orient_loss_func = OrientationLoss

    # load any previous weights
    model_path = os.path.abspath(os.path.dirname(__file__)) + '/weights/'
    latest_model = None
    first_epoch = 0
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    else:
        try:
            latest_model = [x for x in sorted(os.listdir(model_path)) if x.endswith('.pkl')][-1]
        except:
            pass


    if latest_model is not None:
        checkpoint = torch.load(model_path + latest_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt_SGD.load_state_dict(checkpoint['optimizer_state_dict'])
        first_epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        print('Found previous checkpoint: %s at epoch %s'%(latest_model, first_epoch))
        print('Resuming training....')


    model = torch.nn.DataParallel(model)
    total_num_batches = int(len(dataset) / batch_size)

    for epoch in range(first_epoch+1, epochs+1):
        curr_batch = 0
        passes = 0
        for local_batch, local_batch_prev, local_labels in generator:

            truth_orient = Variable(local_labels['Orientation'].float()).cuda()
            truth_conf = Variable(local_labels['Confidence'].long()).cuda()
            truth_dim = Variable(local_labels['Dimensions'].float()).cuda()

            local_batch=Variable(local_batch.float()).cuda()
            [orient, conf, dim] = model(local_batch)

            local_batch_prev=Variable(local_batch_prev.float()).cuda()
            [orient_pre, conf_pre, dim_pre] = model(local_batch_prev)
            
            orient_loss = orient_loss_func(orient, truth_orient, truth_conf)
            dim_loss = dim_loss_func(dim, truth_dim) + dim_loss_func(dim_pre, truth_dim)

            truth_conf = torch.max(truth_conf, dim=1)[1]
            conf_loss = conf_loss_func(conf, truth_conf) + conf_loss_func(conf_pre, truth_conf)

            loss_theta = conf_loss + w * orient_loss
            loss = alpha * dim_loss + loss_theta

            opt_SGD.zero_grad()
            loss.backward()
            opt_SGD.step()


            if passes % 10 == 0:
                print("--- epoch %s | batch %s/%s --- [loss: %s]" %(epoch, curr_batch, total_num_batches, loss.item()))
                passes = 0

            passes += 1
            curr_batch += 1

        # save after every 10 epochs
        if epoch % 10 == 0:
            name = model_path + 'epoch_%s.pkl' % epoch
            print("====================")
            print ("Done with epoch %s!" % epoch)
            print ("Saving weights as %s ..." % name)
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': opt_SGD.state_dict(),
                    'loss': loss
                    }, name)
            print("====================")

if __name__=='__main__':
    main()
