# Monkey-patch because I trained with a newer version.
# This can be removed once PyTorch 0.4.x is out.
# See https://discuss.pytorch.org/t/question-about-rebuild-tensor-v2/14560
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.utils import save_image
import os 
import time
import numpy as np
import cv2
import argparse
import yaml
from torchvision import models
from tqdm import tqdm
from easydict import EasyDict as edict

parser = argparse.ArgumentParser(description='Training code')
parser.add_argument('--config', default='config.yaml', type=str, help='yaml config file')
args = parser.parse_args()
CONFIG = edict(yaml.load(open(args.config, 'r')))
print ('==> CONFIG is: \n', CONFIG, '\n')

if CONFIG.IS_TRAIN:
    LOGDIR = '%s/%s_%d'%(CONFIG.LOGS.LOG_DIR, CONFIG.NAME, int(time.time()))
    SNAPSHOTDIR = '%s/%s_%d'%(CONFIG.LOGS.SNAPSHOT_DIR, CONFIG.NAME, int(time.time()))
    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
    if not os.path.exists(SNAPSHOTDIR):
        os.makedirs(SNAPSHOTDIR)

def to_varabile(arr, requires_grad=False,is_cuda=True):
    if type(arr) == np.ndarray:
        tensor = torch.from_numpy(arr)
    else:
        tensor = arr
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor = tensor.to(device)
    var = Variable(tensor, requires_grad=requires_grad)
    return var

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
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

MEAN_var = to_varabile(np.array(CONFIG.DATASET.MEAN, dtype=np.float32)[:,np.newaxis,np.newaxis], requires_grad=False, is_cuda=True)
        
######################################################################################################################
#                           "Globally and Locally Consistent Image Completion" Model
######################################################################################################################

def AffineAlignOp(features, idxs, aligned_height, aligned_width, Hs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def _transform_matrix(Hs, w, h):
        _Hs = np.zeros(Hs.shape, dtype = np.float32)
        for i, H in enumerate(Hs):
            H0 = np.concatenate((H, np.array([[0, 0, 1]])), axis=0)
            A = np.array([[2.0 / w, 0, -1], [0, 2.0 / h, -1], [0, 0, 1]])
            A_inv = np.array([[w / 2.0, 0, w / 2.0], [0, h / 2.0, h/ 2.0], [0, 0, 1]])
            H0 = A.dot(H0).dot(A_inv)
            H0 = np.linalg.inv(H0)
            _Hs[i] = H0[:-1]
        return _Hs
    bz, C_feat, H_feat, W_feat = features.size()
    N = len(idxs)
    feature_select = features[idxs] # (N, feature_channel, feature_size, feature_size)
    Hs_new = _transform_matrix(Hs, w=W_feat, h=H_feat) # return (N, 2, 3)
    Hs_var = Variable(torch.from_numpy(Hs_new), requires_grad=False).to(device)
    flow = F.affine_grid(theta=Hs_var, size=torch.Size((N, C_feat, H_feat, W_feat))).float().to(device)
    flow = flow[:,:aligned_height, :aligned_width, :]
    rois = F.grid_sample(feature_select.to(device), flow.to(device), mode='bilinear', padding_mode='border') # 'zeros' | 'border'
    return rois
    
def CropAlignOp(feature_var, rois_var, aligned_height, aligned_width, spatial_scale):
    rois_np = rois_var.data.cpu().numpy()
    affinematrixs_feat = []
    for roi in rois_np:
        x1, y1, x2, y2 = roi * float(spatial_scale)
        matrix = np.array([[aligned_width/(x2-x1), 0, -aligned_width/(x2-x1)*x1],
                           [0, aligned_height/(y2-y1), -aligned_height/(y2-y1)*y1]
                          ])
        affinematrixs_feat.append(matrix)
    affinematrixs_feat = np.array(affinematrixs_feat)
    feature_rois = AffineAlignOp(feature_var, np.array(range(rois_var.size(0))), 
                                 aligned_height, aligned_width, affinematrixs_feat)
    return feature_rois


class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.select = ['3', '8', '13', '22', '31']
        self.vgg = models.vgg19(pretrained=True).features

    def forward(self, input):
        features = []
        for name, layer in self.vgg._modules.items():
            input = layer(input)
            if name in self.select:
                features.append(input)
        return features

class ConvBnRelu(nn.Module):
    def __init__(self, inp_dim, out_dim, 
                 kernel_size=3, stride=1, dilation=1, group=1,
                 bias = True, bn = True, relu = True):
        super(ConvBnRelu, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, 0, dilation, group, bias=bias)
        self.pad = nn.ReflectionPad2d((kernel_size-1)//2+(dilation-1))
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ConvBnLeakyrelu(nn.Module):
    def __init__(self, inp_dim, out_dim,
                 kernel_size=3, stride=1, dilation=1, group=1,
                 bias = True, bn = True, relu = True):
        super(ConvBnLeakyrelu, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride,1)
        self.relu = None
        self.bn = None
        if relu:
            self.leakyrelu = nn.LeakyReLU(0.2)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.leakyrelu(x)
        return x
    
class DeconvBnRelu(nn.Module):
    def __init__(self, inp_dim, out_dim, 
                 kernel_size=3, stride=1,
                 bias = True, bn = True, relu = True):
        super(DeconvBnRelu, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.ConvTranspose2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet,self).__init__()
        self.conv = ConvBnRelu(256,256,kernel_size=3,stride=1)

    def forward(self, input):
        x = self.conv(input)
        x = self.conv(x)
        x = x + input
        return x

class GLCIC_G(nn.Module):
    def __init__(self, bias_in_conv=True, pretrainfile=None):
        super(GLCIC_G, self).__init__()
        self.conv1_1 = ConvBnRelu(7, 64, kernel_size=7, stride=1, bias=bias_in_conv)
        self.conv1_2 = ConvBnRelu(64, 128, kernel_size=3, stride=2, bias=bias_in_conv)
        self.conv1_3 = ConvBnRelu(128, 256, kernel_size=3, stride=2, bias=bias_in_conv)
        
        self.resblock2_1 = ResNet()
        self.resblock2_2 = ResNet()
        self.resblock2_3 = ResNet()
        self.resblock2_4 = ResNet()
        
        self.dilatedconv3_1 = ConvBnRelu(256, 256, kernel_size=3, dilation=2, stride=1, bias=bias_in_conv)
        self.dilatedconv3_2 = ConvBnRelu(256, 256, kernel_size=3, dilation=4, stride=1, bias=bias_in_conv)
        self.dilatedconv3_3 = ConvBnRelu(256, 256, kernel_size=3, dilation=8, stride=1, bias=bias_in_conv)
        self.dilatedconv3_4 = ConvBnRelu(256, 256, kernel_size=3, dilation=16, stride=1, bias=bias_in_conv)

        self.resblock4_1 = ResNet()
        self.resblock4_2 = ResNet()
        self.resblock4_3 = ResNet()
        self.resblock4_4 = ResNet()

        self.deconv5_1 = DeconvBnRelu(256, 128, kernel_size=4, stride=2, bias=bias_in_conv)
        self.deconv5_2 = DeconvBnRelu(128, 64, kernel_size=4, stride=2, bias=bias_in_conv)

        self.conv6_1 = nn.Conv2d(64,3,kernel_size=7,stride=1,padding=(7-1)//2,bias=bias_in_conv)
    
    def forward(self, input):
        x = self.conv1_1(input)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.resblock2_1(x)
        x = self.resblock2_2(x)
        x = self.resblock2_3(x)
        x = self.resblock2_4(x)
        x = self.dilatedconv3_1(x)
        x = self.dilatedconv3_2(x)
        x = self.dilatedconv3_3(x)
        x = self.dilatedconv3_4(x)
        x = self.resblock4_1(x)
        x = self.resblock4_2(x)
        x = self.resblock4_3(x)
        x = self.resblock4_4(x)
        x = self.deconv5_1(x)
        x = self.deconv5_2(x)
        x = self.conv6_1(x)
        x = F.tanh(x)
        return x

class GLCIC_Local_D(nn.Module):
    def __init__(self, bias_in_conv=True, pretrainfile=None):
        super(GLCIC_Local_D, self).__init__()
        # local D
        self.local_conv1 = ConvBnLeakyrelu(6, 64, kernel_size=4, stride=2, bias=bias_in_conv, bn=False)
        self.local_conv2 = ConvBnLeakyrelu(64, 128, kernel_size=4, stride=2, bias=bias_in_conv)
        self.local_conv3 = ConvBnLeakyrelu(128, 256, kernel_size=4, stride=2, bias=bias_in_conv)
        self.local_conv4 = ConvBnLeakyrelu(256, 512, kernel_size=4, stride=2, bias=bias_in_conv)
        self.local_conv5 = ConvBnLeakyrelu(512, 1024, kernel_size=4, stride=2, bias=bias_in_conv)
        self.local_conv6 = nn.Conv2d(1024,1,kernel_size=4,stride=1,padding=0)

    def forward(self, input):
        x = self.local_conv1(input)
        x = self.local_conv2(x)
        x = self.local_conv3(x)
        x = self.local_conv4(x)
        x = self.local_conv5(x)
        x = self.local_conv6(x)
        x = F.sigmoid(x)
        return x

class GLCIC_Global_D(nn.Module):
    def __init__(self, bias_in_conv=True, pretrainfile=None):
        super(GLCIC_Global_D, self).__init__()
        # global D
        self.global_conv1 = ConvBnLeakyrelu(6, 64, kernel_size=4, stride=2, bias=bias_in_conv, bn=False)
        self.global_conv2 = ConvBnLeakyrelu(64, 128, kernel_size=4, stride=2, bias=bias_in_conv)
        self.global_conv3 = ConvBnLeakyrelu(128, 256, kernel_size=4, stride=2, bias=bias_in_conv)
        self.global_conv4 = ConvBnLeakyrelu(256, 512, kernel_size=4, stride=2, bias=bias_in_conv)
        self.global_conv5 = ConvBnLeakyrelu(512, 1024, kernel_size=4, stride=2, bias=bias_in_conv)
        self.global_conv6 = ConvBnLeakyrelu(1024, 2048, kernel_size=4, stride=2, bias=bias_in_conv)
        self.global_conv7 = nn.Conv2d(2048,1,kernel_size=4,stride=1,padding=0)


    def forward(self, input):
        x = self.global_conv1(input)
        x = self.global_conv2(x)
        x = self.global_conv3(x)
        x = self.global_conv4(x)
        x = self.global_conv5(x)
        x = self.global_conv6(x)
        x = self.global_conv7(x)
        x = F.sigmoid(x)
        return x

######################################################################################################################
#                                                    Dataset: ATR/LIP
######################################################################################################################

class MyDataset(object):
    def __init__(self, ImageDir,MapDir, istrain=True):
        self.istrain = istrain
        self.imgdir = ImageDir
        self.mapdir = MapDir
        self.imglist = os.listdir(ImageDir)
        self.maplist = os.listdir(MapDir)
        self.imglist.sort()
        self.maplist.sort()
        print ('==> Load Dataset: \n', {'dataset': ImageDir, 'istrain:': istrain, 'len': self.__len__()}, '\n')
        print ('==> Load ParsingDataset: \n', {'dataset': MapDir, 'istrain:': istrain, 'len': self.__len__()}, '\n')
        assert istrain==CONFIG.IS_TRAIN
        
    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, idx):
        return self.loadImage(idx)
    
    def loadImage(self, idx):
        the_choice = np.random.randint(1,100)

        path_img = os.path.join(self.imgdir, self.imglist[idx])
        image = cv2.imread(path_img)
        path_parsing= os.path.join(self.mapdir, self.maplist[idx])
        parsing = cv2.imread(path_parsing)


        if the_choice % 2 == 0:
            image = cv2.flip(image,1)
            parsing = cv2.flip(parsing, 1)

        image = image[:,:,::-1]
        parsing = parsing[:,:,::-1]
        image = cv2.resize(image, (CONFIG.DATASET.INPUT_RES, CONFIG.DATASET.INPUT_RES), interpolation=cv2.INTER_LINEAR)
        parsing = cv2.resize(parsing, (CONFIG.DATASET.INPUT_RES, CONFIG.DATASET.INPUT_RES), interpolation=cv2.INTER_LINEAR)

        parsing = cv2.cvtColor(parsing,cv2.COLOR_BGR2GRAY)
        parsing = np.expand_dims(parsing,axis=2)
        parsing = np.concatenate((parsing,parsing,parsing),axis=2)

        input = (image.astype(np.float32)/255.0 - CONFIG.DATASET.MEAN)

        input = input.transpose(2,0,1)
        parsing = parsing.transpose(2,0,1)
        if self.istrain:
            bbox_c, mask_c = self.randommask(image.shape[0], image.shape[1])
            bbox_d, mask_d = self.randommask(image.shape[0], image.shape[1])
        else:
            if  CONFIG.NAME in ['horse', 'LIP', 'ATR']:
                mask_c = cv2.imread('%s/%s'%(CONFIG.VAL.MASKDIR, self.imglist[idx].replace('jpg', 'png')), cv2.IMREAD_GRAYSCALE).astype(np.float32)
                
            mask_c = cv2.resize(mask_c, (CONFIG.DATASET.INPUT_RES, CONFIG.DATASET.INPUT_RES), interpolation=cv2.INTER_NEAREST)
            mask_c = mask_c[np.newaxis, :,:]
            mask_c[mask_c>=1] = 1.0
            mask_c[mask_c<1] = 0.0
            return np.float32(input), np.float32(mask_c), np.int32(idx)

        return np.float32(input), np.float32(parsing), np.float32(mask_c), bbox_c, np.float32(mask_d), bbox_d
    
    def randommask(self, height, width):
        x1, y1 = np.random.randint(0, CONFIG.DATASET.INPUT_RES - CONFIG.DATASET.LOCAL_RES + 1, 2)
        x2, y2 = np.array([x1, y1]) + CONFIG.DATASET.LOCAL_RES
        w, h = np.random.randint(CONFIG.DATASET.HOLE_MIN, CONFIG.DATASET.HOLE_MAX + 1, 2)
        p1 = x1 + np.random.randint(0, CONFIG.DATASET.LOCAL_RES - w)
        q1 = y1 + np.random.randint(0, CONFIG.DATASET.LOCAL_RES - h)
        p2 = p1 + w
        q2 = q1 + h
        mask = np.zeros((height, width), dtype=np.float32)
        mask[q1:q2 + 1, p1:p2 + 1] = 1.0
        bbox = np.array([x1, y1, x1+CONFIG.DATASET.LOCAL_RES, y1+CONFIG.DATASET.LOCAL_RES], dtype=np.int32)
        return bbox, mask[np.newaxis, :,:]



######################################################################################################################
#                                                   Training
######################################################################################################################
def train(dataLoader, model_G,model_Local_D, model_Global_D,vgg, epoch):
    LOSS = nn.BCELoss()
    batch_time = AverageMeter('batch_time')
    data_time = AverageMeter('data_time')
    losses_G = AverageMeter('losses_G')
    losses_D = AverageMeter('losses_D')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # switch to train mode
    model_G.train()
    model_Global_D.train()
    model_Local_D.train()

    end = time.time()
    for i, data in enumerate(dataLoader):
        j = i + 1

        data_time.update(time.time() - end)

        input3ch, Pmap, mask_c, bbox_c, mask_d, bbox_d = data

        input7ch = torch.cat([input3ch * (1 - mask_c),Pmap , mask_c], dim=1)
        
        input3ch_var = to_varabile(input3ch, requires_grad=True, is_cuda=True) + MEAN_var
        input7ch_var = to_varabile(input7ch, requires_grad=True, is_cuda=True)
        bbox_c_var = to_varabile(bbox_c, requires_grad=False, is_cuda=True)
        mask_c_var = to_varabile(mask_c, requires_grad=True, is_cuda=True)
        Pmap_var = to_varabile(Pmap, requires_grad=True, is_cuda=True)

        out_G = model_G(input7ch_var)

        target_feature = vgg(input3ch_var)
        output_feature = vgg(out_G)
        loss_P = 0
        for f1,f2 in zip(target_feature,output_feature):
            loss_P += torch.mean((f1-f2) ** 2)

        completion = (input3ch_var)*(1 - mask_c_var) + out_G * mask_c_var

        completion = completion.to(device)

        local_completion = CropAlignOp(completion, bbox_c_var, 
                                       CONFIG.DATASET.LOCAL_RES, CONFIG.DATASET.LOCAL_RES, spatial_scale=1.0)
        local_input3ch = CropAlignOp(input3ch_var, bbox_c_var, 
                                       CONFIG.DATASET.LOCAL_RES, CONFIG.DATASET.LOCAL_RES, spatial_scale=1.0)
        local_parsing = CropAlignOp(Pmap, bbox_c_var,
                                       CONFIG.DATASET.LOCAL_RES, CONFIG.DATASET.LOCAL_RES, spatial_scale=1.0)


        local_completion6ch = torch.cat([local_completion, local_parsing], dim=1)
        completion6th = torch.cat([completion, Pmap_var], dim=1)
        local_real6th = torch.cat([local_input3ch, local_parsing], dim=1)
        real6th = torch.cat([input3ch_var, Pmap_var], dim=1)

        out_D_local_fake = model_Local_D(local_completion6ch)
        loss_D_local_fake = LOSS(out_D_local_fake, torch.zeros_like(out_D_local_fake))

        out_D_local_real = model_Local_D(local_real6th)
        loss_D_local_real = LOSS(out_D_local_real, torch.ones_like(out_D_local_real))

        out_D_global_fake = model_Global_D(completion6th)
        loss_D_global_fake = LOSS(out_D_global_fake, torch.zeros_like(out_D_global_fake))

        out_D_global_real = model_Global_D(real6th)
        loss_D_global_real = LOSS(out_D_global_real, torch.ones_like(out_D_global_real))

        loss_local_D = loss_D_local_fake + loss_D_local_real
        loss_global_D = loss_D_global_fake + loss_D_global_real

        optimizer_local_D = torch.optim.Adam(model_Local_D.parameters(), CONFIG.SOLVER.LR,
                                                 weight_decay=CONFIG.SOLVER.WEIGHTDECAY)
        optimizer_global_D = torch.optim.Adam(model_Global_D.parameters(), CONFIG.SOLVER.LR,
                                                  weight_decay=CONFIG.SOLVER.WEIGHTDECAY)
        optimizer_G = torch.optim.Adam(model_G.parameters(), CONFIG.SOLVER.LR,
                                           weight_decay=CONFIG.SOLVER.WEIGHTDECAY)


        loss_D = loss_local_D * CONFIG.SOLVER.LAMBDA_L + loss_global_D * CONFIG.SOLVER.LAMBDA_G
        loss_G = loss_P * CONFIG.SOLVER.LAMBDA_P

        losses_D.update(loss_D.data[0], input3ch.size(0))
        losses_G.update(loss_G.data[0], input3ch.size(0))

        optimizer_G.zero_grad()
        loss_G.backward(retain_graph=True)
        optimizer_G.step()

        optimizer_local_D.zero_grad()
        optimizer_global_D.zero_grad()
        loss_D.backward()
        optimizer_local_D.step()
        optimizer_global_D.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if j % CONFIG.LOGS.PRINT_FREQ == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  #'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'G {loss_G.val:.4f}({loss_G.avg:.4f})\t'
                  'D {loss_D.val:.4f}({loss_D.avg:.4f})\t'.format(
                   epoch, j, len(dataLoader), batch_time=batch_time, #data_time=data_time,
                   loss_G=losses_G, loss_D=losses_D,))
        
        if j % CONFIG.LOGS.LOG_FREQ == 0:
            vis = torch.cat([input3ch_var * (1 - mask_c_var),completion],dim = 0)
            save_image(vis.data, os.path.join(LOGDIR, 'epoch%d_%d_vis.jpg'%(epoch, j)), nrow=input3ch.size(0), padding=2,
                       normalize=False, range=None, scale_each=True, pad_value=0)
            
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vgg = VGGNet().to(device).eval()
    BATCHSIZE = 1
    dataset_ATR = MyDataset(ImageDir=CONFIG.DATASET.TRAINDIR_ATR,MapDir=CONFIG.DATASET.MAPDIR_ATR, istrain=True)
    dataLoader_ATR = torch.utils.data.DataLoader(dataset_ATR, batch_size=BATCHSIZE, shuffle=False, num_workers=CONFIG.SOLVER.WORKERS, pin_memory=False)
    dataset_LIP = MyDataset(ImageDir=CONFIG.DATASET.TRAINDIR_LIP, MapDir=CONFIG.DATASET.MAPDIR_LIP, istrain=True)
    dataLoader_LIP = torch.utils.data.DataLoader(dataset_LIP, batch_size=BATCHSIZE, shuffle=False,
                                                 num_workers=CONFIG.SOLVER.WORKERS, pin_memory=False)

    model_G = GLCIC_G(bias_in_conv=True).to(device)
    model_Local_D = GLCIC_Local_D(bias_in_conv=True).to(device)
    model_Global_D = GLCIC_Global_D(bias_in_conv=True).to(device)

    epoches = CONFIG.TOTAL_EPOCHES_ATR
    for epoch in range(epoches):
        Epoch = epoch + 1
        print ('===========>   [Epoch %d] training    <==========='%Epoch)
        train(dataLoader_ATR, model_G, model_Local_D,model_Global_D,vgg, epoch)
        if epoch % CONFIG.LOGS.SNAPSHOT_FREQ == 0 :
            torch.save(model_G.state_dict(), os.path.join(SNAPSHOTDIR, '_0_G_%d.pkl'%(Epoch)))
            torch.save(model_Local_D.state_dict(), os.path.join(SNAPSHOTDIR, '_0_D_Local_%d.pkl'%(Epoch)))
            torch.save(model_Global_D.state_dict(), os.path.join(SNAPSHOTDIR, '_0_D_Global_%d.pkl' % (Epoch)))

    #torch.save(model_G.state_dict(), os.path.join(SNAPSHOTDIR, '_0_G.pkl')
    #torch.save(model_Local_D.state_dict(), os.path.join(SNAPSHOTDIR, '_0_D_Local.pkl')
    #torch.save(model_Global_D.state_dict(), os.path.join(SNAPSHOTDIR, '_0_D_Global.pkl')

    epoches = CONFIG.TOTAL_EPOCHES_LIP
    for epoch in range(epoches):
        Epoch = epoch + 1
        print('===========>   [Epoch %d] training    <===========' % Epoch)
        train(dataLoader_LIP, model_G, model_Local_D, model_Global_D, vgg, epoch)
        if epoch % CONFIG.LOGS.SNAPSHOT_FREQ == 0:
            torch.save(model_G.state_dict(), os.path.join(SNAPSHOTDIR, '_1_G_%d.pkl' % (Epoch)))
            torch.save(model_Local_D.state_dict(), os.path.join(SNAPSHOTDIR, '_1_D_Local_%d.pkl' % (Epoch)))
            torch.save(model_Global_D.state_dict(), os.path.join(SNAPSHOTDIR, '_1_D_Global_%d.pkl' % (Epoch)))

    #torch.save(model_G.state_dict(), os.path.join(SNAPSHOTDIR, '_1_G.pkl')
    #torch.save(model_Local_D.state_dict(), os.path.join(SNAPSHOTDIR, '_1_D_Local.pkl')
    #torch.save(model_Global_D.state_dict(), os.path.join(SNAPSHOTDIR, '_1_D_Global.pkl')

    
def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from blend import blend
    if not os.path.exists(CONFIG.VAL.OUTDIR):
        os.makedirs(CONFIG.VAL.OUTDIR)
    
    dataset = MyDataset(ImageDir=CONFIG.DATASET.VALDIR, istrain=False)
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    
    model_G = GLCIC_G(bias_in_conv=True, pretrainfile=CONFIG.VAL.INIT).to(device)
        
    # switch to eval mode
    model_G.eval()

    for data in tqdm(dataLoader):

        input3ch, mask_c, idxs = data
        filename = dataset.imglist[idxs.numpy()[0]]
        input4ch = torch.cat([input3ch * (1 - mask_c), mask_c], dim=1)
        
        input3ch_var = to_varabile(input3ch, requires_grad=False, is_cuda=True) + MEAN_var
        input4ch_var = to_varabile(input4ch, requires_grad=False, is_cuda=True)
        mask_c_var = to_varabile(mask_c, requires_grad=False, is_cuda=True)
        
        out_G = model_G(input4ch_var)        
        completion = (input3ch_var)*(1 - mask_c_var) + out_G * mask_c_var
        
        completion_np = completion.data.cpu().numpy().transpose((0, 2, 3, 1))[0] *255.0
        
        path = os.path.join(dataset.imgdir, filename)
        
        image = cv2.imread(path)[:,:,::-1]
        completion_np = cv2.resize(completion_np, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        completion_np = np.uint8(completion_np)
        
        mask = cv2.imread('%s/%s'%(CONFIG.VAL.MASKDIR, filename.replace('jpg', 'png')))
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        completion_np[mask<0.5] = image[mask<0.5]
        
        cv2.imwrite('%s/%s'%(CONFIG.VAL.OUTDIR, filename), np.uint8(completion_np[:,:,::-1]))

                
    
    
if __name__ == '__main__':
    if CONFIG.IS_TRAIN:
        main()
    else:
        test()
    
    
    
