import cloudpickle
import torch
from torchvision.utils import save_image
import numpy as np
import cv2

from GAN import Generator


#取り出すepochを指定する
point = 100

#モデルの構造を定義
z_dim = 30
num_class = 49
G = Generator(z_dim = z_dim, num_class = num_class)
#checkpointを取り出す
checkpoint = torch.load('./checkpoint_cGAN/G_model_{}'.format(point), map_location=torch.device('cpu'))

#Generatorにパラメータを入れる
G.load_state_dict(checkpoint['model_state_dict'])

#検証モードにしておく
G.eval()

#pickleで保存
with open ('KMNIST_cGAN.pkl','wb')as f:
    cloudpickle.dump(G,f)

letter = 'あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわゐゑをんゝ'

strs = input()
with open('KMNIST_cGAN.pkl','rb')as f:
    Generator = cloudpickle.load(f)

for i in range(len(str(strs))):
    noise = torch.normal(mean = 0.5, std = 0.2, size = (1, 30))
    str_index = letter.index(strs[i])
    tmp = np.identity(49)[str_index]
    tmp = np.array(tmp, dtype = np.float32)
    label = [tmp]

    img = Generator(noise, torch.Tensor(label))
    img = img.reshape((28,28))
    img = img.detach().numpy().tolist()

    if i == 0:
        comp_img = img
    else:
        comp_img.extend(img)

save_image(torch.tensor(comp_img), './sentence.png', nrow=len(str(strs)))