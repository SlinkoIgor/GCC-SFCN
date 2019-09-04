from matplotlib import pyplot as plt
import os
import random
import torch
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import misc.transforms as own_transforms
import pandas as pd

from models.CC import CrowdCounter
from config import cfg
from misc.utils import *
import scipy.io as sio
from PIL import Image, ImageOps

from tqdm import tqdm

# torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

exp_name = './vis'
if not os.path.exists(exp_name):
    os.mkdir(exp_name)

mean_std = cfg.DATA.MEAN_STD
img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
restore = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])
pil_to_tensor = standard_transforms.ToTensor()

dataRoot = '.' #'/media/D/DataSet/CC/UCF-qnrf/768x1024_1221/test'

model_path = './ThisRepo_mae_99.1_mse_185.4.pth'
def main():
    # file_list = [filename for filename in os.listdir(dataRoot+'/img/') if os.path.isfile(os.path.join(dataRoot+'/img/',filename))]
    file_list = [filename for root,dirs, filename in os.walk(dataRoot+'/img/')]
    # pdb.set_trace()

    ht_img = cfg.TRAIN.INPUT_SIZE[0]
    wd_img = cfg.TRAIN.INPUT_SIZE[1]

    test(file_list[0], model_path)


def test(file_list, model_path):

    f_out = open('report.txt', 'w')

    net = CrowdCounter()
    net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    # net = tr_net.CNN()
    # net.load_state_dict(torch.load(model_path))
    net.eval()

    maes = []
    mses = []

    for filename in tqdm(file_list):
        imgname = dataRoot + '/img/' + filename
        filename_no_ext = filename.split('.')[0]

        # denname = dataRoot + '/den/' + filename_no_ext + '.csv'


        # den = pd.read_csv(denname, sep=',',header=None).values
        # den = den.astype(np.float32, copy=False)

        try:
            img = Image.open(imgname)
        except Exception as e:
            print(e)
            continue

        if img.mode == 'L':
            img = img.convert('RGB')

        # prepare
        wd_1, ht_1 = img.size
        # pdb.set_trace()

        if wd_1 < cfg.DATA.STD_SIZE[1]:
            dif = cfg.DATA.STD_SIZE[1] - wd_1
            img = ImageOps.expand(img, border=(0,0,dif,0), fill=0)
            pad = np.zeros([ht_1,dif])
            # den = np.array(den)
            # den = np.hstack((den,pad))

        if ht_1 < cfg.DATA.STD_SIZE[0]:
            dif = cfg.DATA.STD_SIZE[0] - ht_1
            img = ImageOps.expand(img, border=(0,0,0,dif), fill=0)
            pad = np.zeros([dif,wd_1])
            # den = np.array(den)
            # den = np.vstack((den,pad))

        img = img_transform(img)

        # gt = np.sum(den)

        img = torch.Tensor(img[None,:,:,:])

        #forward
        pred_map = net.test_forward(img)

        pred_map = pred_map.cpu().data.numpy()[0,0,:,:] / 100.
        pred = np.sum(pred_map)
        print(filename, pred, pred_map.max(), file=f_out)

        # maes.append(abs(pred-gt))
        # mses.append((pred-gt)*(pred-gt))

        np.save(f'preds/pred_map_{filename_no_ext}_{str(float(pred))}.npy', pred_map/100.0)


        # vis
        # pred_map = pred_map/np.max(pred_map+1e-20)
        pred_map = pred_map[0:ht_1,0:wd_1]


        # den = den/np.max(den+1e-20)
        # den = den[0:ht_1,0:wd_1]

        # den_frame = plt.gca()
        # # plt.imshow(den, 'jet')
        # den_frame.axes.get_yaxis().set_visible(False)
        # den_frame.axes.get_xaxis().set_visible(False)
        # den_frame.spines['top'].set_visible(False)
        # den_frame.spines['bottom'].set_visible(False)
        # den_frame.spines['left'].set_visible(False)
        # den_frame.spines['right'].set_visible(False)
        # plt.savefig(exp_name+'/'+filename_no_ext+'_gt_'+str(int(gt))+'.png',\
        #     bbox_inches='tight',pad_inches=0,dpi=150)

        # plt.close()

        # sio.savemat(exp_name+'/'+filename_no_ext+'_gt_'+str(int(gt))+'.mat',{'data':den})

        plt.imshow(pred_map)
        plt.colorbar()
        plt.savefig(exp_name+'/'+filename_no_ext+'_pred_'+str(float(pred))+'.png',\
            bbox_inches='tight',pad_inches=0,dpi=150)
        plt.close()

        # sio.savemat(exp_name+'/'+filename_no_ext+'_pred_'+str(float(pred))+'.mat',{'data':pred_map})

        # diff = den-pred_map

        # diff_frame = plt.gca()
        # plt.imshow(diff, 'jet')
        # plt.colorbar()
        # diff_frame.axes.get_yaxis().set_visible(False)
        # diff_frame.axes.get_xaxis().set_visible(False)
        # diff_frame.spines['top'].set_visible(False)
        # diff_frame.spines['bottom'].set_visible(False)
        # diff_frame.spines['left'].set_visible(False)
        # diff_frame.spines['right'].set_visible(False)
        # plt.savefig(exp_name+'/'+filename_no_ext+'_diff.png',\
        #     bbox_inches='tight',pad_inches=0,dpi=150)

        # plt.close()

        # sio.savemat(exp_name+'/'+filename_no_ext+'_diff.mat',{'data':diff})

        # print('[file %s]: [pred %.2f], [gt %.2f]' % (filename, pred, gt))
    # print(np.average(np.array(maes)))
    # print(np.sqrt(np.average(np.array(mses))))
    f_out.close()




if __name__ == '__main__':
    main()




