import os
import numpy as np 
import math
import cv2

import time

start = time.clock()

def psnr(img1, img2):
    mse = np.mean((img1/1-img2/1)**2)    # 15.0*15.0/mse 分母在除法时会倒上去  所以也可以写成 img1/255
    if mse < 1.0e-10:
        return 100*1.0
    #return 10*math.log10(255.0*255.0/mse)
    return 10*math.log10(255.0*255.0/mse)

def mse(img1,img2):
    mse = np.mean((img1/1-img2/1)**2)
    return mse

def ssim(y_true , y_pred):
    u_true = np.mean(y_true)
    u_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    std_true = np.sqrt(var_true)
    std_pred = np.sqrt(var_pred)
    c1 = np.square(0.01*7)
    c2 = np.square(0.03*7)

    ssim = (2*u_true*u_pred+c1) * (2*std_pred*std_true+c2)
    denom = (u_true**2 + u_pred**2 +c1)*(var_pred+var_true+c2)
    return ssim/denom

#path1 = '/media/dew/Samsung_T5/数据/Kodak/arcnn_4bits_q10/'
#path2 = '/media/dew/Samsung_T5/数据/Kodak/4bits_q10/'
#path1 = 'G:/data/Kodak/6bits/'       #opencv不接受non-ascii的路径 所以读不到中文路径，①有方法解决 ②干脆把路径名改为英文。
#path2 = 'G:/data/Kodak/varnn_6bits_q20/'
path = 'E:/BestTools/文献管理/ReadPapers/MyOverleafRepo/ICME 2020London/图片/比较/72-197'
img_a = cv2.imdecode(np.fromfile(os.path.join(path, '72-197.png'), dtype=np.uint8),  cv2.IMREAD_COLOR)
img_b = cv2.imdecode(np.fromfile(os.path.join(path, 'ours.jpg'), dtype=np.uint8),  cv2.IMREAD_COLOR)

psnr_ours = psnr(img_a, img_b)
ssim_ours = ssim(img_a, img_b)

print("ours PSNR:", psnr_ours)
print("ours_ssim:", ssim_ours)
'''
list_psnr = []
list_ssim = []
list_mse = []
'''
#path = "G:\数据\Kodak\varnn_6bits_q20\"
'''
path_list2 = os.listdir(path2)
for filename in path_list2:
    #print(os.path.join(path2,filename))
    img_a = cv2.imread(os.path.join(path2,filename))
    #print(img_a)
    img_b = cv2.imread(os.path.join(path1,filename))
    psnr_num = psnr(img_a, img_b)
    ssim_num = ssim(img_a, img_b)
    mse_num = mse(img_a, img_b)
    list_psnr.append(psnr_num)
    list_ssim.append(ssim_num)
    list_mse.append(mse_num)

print("平均PSNR:", np.mean(list_psnr))
print("平均ssim:", np.mean(list_ssim))
print("平均mse:", np.mean(list_mse))
'''


elapsed = (time.clock() - start)
print("Time used:" , elapsed)







