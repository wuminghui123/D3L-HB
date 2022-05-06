import torch
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第二块GPU（从0开始）
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import random
import torch.nn.functional as F
import torchvision
import numpy as np
from math import *
import matplotlib.pyplot as plt
from torch.autograd import Variable
from IPython import display
import torch.utils.data as Data
import torch.nn as nn
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from scipy.linalg import block_diag
import datetime
from torch.nn.utils import *

Nc = 32 #number of subcarriers
N = 2   # Number of paths
Nt = 64 # Number of Antennas at the BS
Nr = 1  # Number of Antennas at the UE
# B = 30

L = 8   # number of pilot OFDM symbols
SNR_dB = 10  # SNR
K = 2   # number of UEs
snr =  10**(SNR_dB/10)/K


def Num2Bit(Num, B):
    Num_ = Num.type(torch.uint8)

    def integer2bit(integer, num_bits=B * 2):
        dtype = integer.type()
        exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
        exponent_bits = exponent_bits.repeat(integer.shape + (1,))
        out = integer.unsqueeze(-1) // 2 ** exponent_bits
        return (out - (out % 1)) % 2

    bit = integer2bit(Num_)
    bit = (bit[:, :, B:]).reshape(-1, Num_.shape[1] * B)
    return bit.type(torch.float32)


def Bit2Num(Bit, B):
    Bit_ = Bit.type(torch.float32)
    Bit_ = torch.reshape(Bit_, [-1, int(Bit_.shape[1] / B), B])
    num = torch.zeros(Bit_[:, :, 0].shape).cuda()
    for i in range(B):
        num = num + Bit_[:, :, i] * 2 ** (B - 1 - i)
    return num


class Quantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = torch.round(x * step - 0.5)
        out = Num2Bit(out, B)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of constant arguments to forward must be None.
        # Gradient of a number is the sum of its four bits.
        b, _ = grad_output.shape
        grad_num = torch.sum(grad_output.reshape(b, -1, ctx.constant), dim=2)
        return grad_num, None


class Dequantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = Bit2Num(x, B)
        out = (out + 0.5) / step
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        # repeat the gradient of a Num for four time.
        #b, c = grad_output.shape
        #grad_bit = grad_output.repeat(1, 1, ctx.constant)
        grad_bit = grad_output.repeat_interleave(ctx.constant,dim=1)
        return grad_bit, None


class QuantizationLayer(nn.Module):

    def __init__(self, B):
        super(QuantizationLayer, self).__init__()
        self.B = B

    def forward(self, x):
        out = Quantization.apply(x, self.B)
        return out


class DequantizationLayer(nn.Module):

    def __init__(self, B):
        super(DequantizationLayer, self).__init__()
        self.B = B

    def forward(self, x):
        out = Dequantization.apply(x, self.B)
        return out


class MyLoss_OFDM(torch.nn.Module):   #输入是信道和E2E网络的输出预编码,输出是频谱效率
    def __init__(self):
        super(MyLoss_OFDM, self).__init__()
    def forward(self, H0, out, parm_set):#H0第0个维度是样本 第1个维度是用户，第2个维度是子载波，第3个维度是天线
        Nc  = parm_set[0]
        Nt  = parm_set[1]
        Nr  = parm_set[2]
        snr = parm_set[3]
        B   = parm_set[4]
        K   = parm_set[5]
        
        H = H0.permute(0,2,1,3)#H第0个维度是样本 第1个维度是子载波，第2个维度是用户，第3个维度是天线
        num = out.shape[0]
        Nc = H.shape[1]
        H_real = H[:,:,:,0:Nt] 
        H_imag = H[:,:,:,Nt:2*Nt]
        Hs = torch.zeros([num,Nc,K*2,Nt*2])
        Hs = Hs.cuda()
        Hs[:,:,0:K,0:Nt] = H_real
        Hs[:,:,K:2*K,Nt:2*Nt] = H_real
        Hs[:,:,0:K,Nt:2*Nt] = H_imag
        Hs[:,:,K:2*K,0:Nt] = -H_imag
        
        F = torch.zeros([num,Nc,Nt*2,K*2])
        F = F.cuda()
        F[:,:,0:Nt,0:K] = out[:,0:K*Nt*Nc].reshape(num,Nc,Nt,K)
        F[:,:,Nt:2*Nt,K:2*K] = out[:,0:K*Nt*Nc].reshape(num,Nc,Nt,K)
        F[:,:,0:Nt,K:2*K] = out[:,K*Nt*Nc:2*K*Nt*Nc].reshape(num,Nc,Nt,K)
        F[:,:,Nt:2*Nt,0:K] = -out[:,K*Nt*Nc:2*K*Nt*Nc].reshape(num,Nc,Nt,K)     
        R = 0
        Hk = torch.matmul(Hs,F)
        noise = 1/snr
        for i in range(K):
            signal = Hk[:,:,i,i]*Hk[:,:,i,i]+Hk[:,:,i,i+K]*Hk[:,:,i,i+K]
            interference = torch.zeros(num,Nc)
            interference = interference.cuda()
            for j in range(K):
                    if j != i:
                        interference = interference + Hk[:,:,i,j]*Hk[:,:,i,j] + Hk[:,:,i,j+K]*Hk[:,:,i,j+K]
            SINR = signal/(noise+interference)
            R = R+torch.sum(torch.log2(1+SINR))
        R = -R/num/Nc
        return R



def mish(x):
    x = x * (torch.tanh(F.softplus(x)))
    return x
class Mish(nn.Module):
    def __init__(self):
        super().__init__()
#         print("Mish activation loaded...")
    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))
        return x
class RES_BLOCK(nn.Module): #输入信道输出 量化后的B比特反馈信息
    def __init__(self,channel_list):
        super(RES_BLOCK,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = channel_list[0], #图片通道数
                out_channels = channel_list[1],#filter数量
                kernel_size = (5,1), #filter大小
                stride=1,  #扫描步进
                padding=(2,0), #周围围上2圈0 使得输出的宽和高和之前一样不变小
            ),
            nn.BatchNorm2d(channel_list[1], eps=1e-05, momentum=0.1, affine=True),
            Mish(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels = channel_list[1], #图片通道数
                out_channels = channel_list[2],#filter数量
                kernel_size = (5,1), #filter大小
                stride=1,  #扫描步进
                padding=(2,0), #周围围上2圈0 使得输出的宽和高和之前一样不变小
            ),
            nn.BatchNorm2d(channel_list[2], eps=1e-05, momentum=0.1, affine=True),
            Mish(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels = channel_list[2], #图片通道数
                out_channels = channel_list[0],#filter数量
                kernel_size = (5,1), #filter大小
                stride=1,  #扫描步进
                padding=(2,0), #周围围上2圈0 使得输出的宽和高和之前一样不变小
            ),
            nn.BatchNorm2d(channel_list[0], eps=1e-05, momentum=0.1, affine=True),
        )
        
        
    def forward(self,x_ini):
        
        
        x = self.conv1(x_ini)
        x = self.conv2(x)
        x = self.conv3(x)
        x = mish(x+x_ini)
        return x

def DFT_matrix(N):
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp( - 2 * pi * 1J / N )
    W = np.power( omega, i * j ) / sqrt(N)
    return np.mat(W)
Nc = 32
W = DFT_matrix(Nc)
W_real = torch.from_numpy(np.real(W)).cuda()
W_imag = torch.from_numpy(np.imag(W)).cuda()
W_real = W_real.float()
W_imag = W_imag.float()


class DNN_US_RF_OFDM(nn.Module): #包含BS端的导频矩阵，经过信道H，到达UE后被压缩量化输出反馈比特
    def __init__(self,parm_set):
        Nc  = parm_set[0]
        Nt  = parm_set[1]
        Nr  = parm_set[2]
        snr = parm_set[3]
        B   = parm_set[4]
        K   = parm_set[5]
        super(DNN_US_RF_OFDM,self).__init__()
        

        self.pilot = nn.Linear(Nt,L,bias=False)#全连接

        self.res  = RES_BLOCK([2*L,256,512])
        
        self.FC2 = nn.Linear(2*Nc*L,1024)#全连接
        self.bn2 = nn.BatchNorm1d(1024)
        self.relu2 = nn.ReLU()
        
        
        self.FC3 = nn.Linear(1024,512)#全连接
        self.bn3 = nn.BatchNorm1d(512)
        self.relu3 = nn.ReLU()
        
        self.FC4 = nn.Linear(512,B)#全连接
        self.bn4 = nn.BatchNorm1d(B)

        
        self.QL = QuantizationLayer(1)
        #最后用sigm
        
    def forward(self,h, parm_set):
        Nc  = parm_set[0]
        Nt  = parm_set[1]
        Nr  = parm_set[2]
        snr = parm_set[3]
        B   = parm_set[4]
        K   = parm_set[5]
        
        h_real = h[:,:,0:Nt].reshape(-1,Nc,Nt,1)
        h_imag = h[:,:,Nt:2*Nt].reshape(-1,Nc,Nt,1)
        
        F_real = torch.cos(self.pilot.weight)/sqrt(Nt)*sqrt(K)
        F_imag = torch.sin(self.pilot.weight)/sqrt(Nt)*sqrt(K)
        L_real = (torch.matmul(F_real,h_real)-torch.matmul(F_imag,h_imag)).reshape(-1,1,Nc,L)
        L_imag = (torch.matmul(F_real,h_imag)+torch.matmul(F_imag,h_real)).reshape(-1,1,Nc,L)  
        L_sum = torch.cat((L_real,L_imag), 1)
        
        num = h.shape[0]
        noise = torch.randn(num,2,Nc,L)/sqrt(2*snr)
        noise = noise.cuda()
        L_sum = L_sum + noise
        #noise = (np.random.randn(L,1)+1j*np.random.randn(L,1))/sqrt(2*snr)
        
        L_real = (torch.matmul(W_real,L_sum[:,0,:,:])-torch.matmul(W_imag,L_sum[:,1,:,:])).reshape(-1,Nc,L)
        L_imag = (torch.matmul(W_imag,L_sum[:,0,:,:])+torch.matmul(W_real,L_sum[:,1,:,:])).reshape(-1,Nc,L)
        L_sum = torch.cat((L_real,L_imag), 2)
        
        x = L_sum.transpose(1,2)
        x = x.reshape(-1,2*L,Nc,1)
        x = self.res(x)
        x = x.reshape(-1,2*L*Nc)
        
        
        x = self.FC2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        
        x = self.FC3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.FC4(x)
        x = self.bn4(x)
        x = torch.sigmoid(x)
        x = self.QL(x)
        return x

class DNN_BS_hyb_OFDM(nn.Module): #根据所有用户的反馈比特，输出混合预编码矩阵
    def __init__(self,parm_set):
        Nc  = parm_set[0]
        Nt  = parm_set[1]
        Nr  = parm_set[2]
        snr = parm_set[3]
        B   = parm_set[4]
        K   = parm_set[5]
        super(DNN_BS_hyb_OFDM,self).__init__()
        
        self.DQL = DequantizationLayer(1)
                
        self.FC1 = nn.Linear(K*B,2048)#全连接
        self.bn1 = nn.BatchNorm1d(2048)
        self.mish1 = Mish()
        
        self.FC2 = nn.Linear(2048,1024)#全连接
        self.bn2 = nn.BatchNorm1d(1024)
        self.mish2 = Mish()
        
        self.FC3 = nn.Linear(1024,2*K*K*Nc+K*Nt)#全连接
        self.bn3 = nn.BatchNorm1d(2*K*K*Nc)
        self.mish3 = Mish()
        
        self.res1  = RES_BLOCK([2*K*K,256,512])
        self.conv =  nn.Conv2d(
                        in_channels = 2*K*K, #图片通道数
                        out_channels = 2*K*K,#filter数量
                        kernel_size = (5,1), #filter大小
                        stride=1,  #扫描步进
                        padding=(2,0), #周围围上2圈0 使得输出的宽和高和之前一样不变小
                        )
        #最后用sigm
        
    def forward(self,x, parm_set):
        Nc  = parm_set[0]
        Nt  = parm_set[1]
        Nr  = parm_set[2]
        snr = parm_set[3]
        B   = parm_set[4]
        K   = parm_set[5]
        
        x = self.DQL(x)-0.5
        
        x = self.FC1(x)
        x = self.bn1(x)
        x = self.mish1(x)
        
        x = self.FC2(x)
        x = self.bn2(x)
        x = self.mish2(x)
        
        
        x_ini = self.FC3(x)
        
        RF_real = torch.cos(x_ini[:,2*K*K*Nc:(2*K*K*Nc+K*Nt)]).reshape(-1,Nt,K)/sqrt(Nt)
        RF_imag = torch.sin(x_ini[:,2*K*K*Nc:(2*K*K*Nc+K*Nt)]).reshape(-1,Nt,K)/sqrt(Nt)
        
        x = x_ini[:,0:2*K*K*Nc]
        x = self.bn3(x)
        x = self.mish3(x)
        
        x = x.reshape(-1,2*K*K,Nc,1)
        x = self.res1(x)
        x = self.conv(x)
        
        x = x.transpose(1,2)
        
        BB_real = x[:,:,0:K*K,0].reshape(-1,Nc,K,K)
        BB_imag = x[:,:,K*K:2*K*K,0].reshape(-1,Nc,K,K)
        BB_real = BB_real.permute(1,0,2,3)
        BB_imag = BB_imag.permute(1,0,2,3)
        
        F_real = (torch.matmul(RF_real,BB_real)-torch.matmul(RF_imag,BB_imag)).reshape(Nc,-1,K*Nt)
        F_imag = (torch.matmul(RF_imag,BB_real)+torch.matmul(RF_real,BB_imag)).reshape(Nc,-1,K*Nt)
        F_real = F_real.permute(1,0,2)
        F_imag = F_imag.permute(1,0,2)
        F = torch.cat((F_real,F_imag), 2)
        
        F_sigma = torch.sqrt(torch.sum(F*F,[2]))
        sigma2 = torch.FloatTensor([sqrt(K)]).cuda()
        F = F/F_sigma.reshape(-1,Nc,1)*torch.min(F_sigma,sigma2).reshape(-1,Nc,1)
        
        F_real = F[:,:,0:K*Nt].reshape(-1,K*Nt*Nc)
        F_imag = F[:,:,K*Nt:2*K*Nt].reshape(-1,K*Nt*Nc)
        F = torch.cat((F_real,F_imag), 1)

        
        return F