import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
from PIL import Image
import os
import pickle
import h5py
import os.path as osp
this_dir = os.path.dirname(os.path.realpath((__file__)))
data_folder = osp.join(this_dir, 'dataset_2014')
image_path = '/ssddata/s1985335/raid/IC-GAN/coco/test2014/COCO_val2014_000000016491.jpg'

val_imgid = pickle.load(open(os.path.join(data_folder,'val_ids.pkl'),'rb'))
val_bottomupregions = h5py.File(os.path.join(data_folder,'val36.hdf5'),'r')
val_bb = list(val_bottomupregions['image_bb'])

test_img_id = str.split(image_path,'/')[-1]
test_id = int(str.split(test_img_id,'_')[-1][:-4])

test_idx = [test_id==idx for idx in val_imgid].index(True)
test_bb = val_bb[test_idx]


print(test_bb, test_idx)

image = Image.open(image_path)

# fig, ax = plt.subplots(figsize=(12, 12))
# ax.imshow(image, aspect='equal')
# for bbox in test_bb:
# # bbox = test_bb[10]
#     ax.add_patch(
#         plt.Rectangle((bbox[0], bbox[1]),
#                         bbox[2] - bbox[0],
#                         bbox[3] - bbox[1], fill=False,
#                         edgecolor='red', linewidth=2.0)
#         )

# plt.axis('off')
# plt.tight_layout()
# plt.draw()
# plt.show()

words = ['a','number', 'of', 'baseball', 'players', 'on', 'a', 'field']
# words = [rev_word_map[ind] for ind in seq]
alphas = F.softmax(torch.randn(len(words),36), dim=1)

for t in range(len(words)):
    if t > 50:
        break
    ax = plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

    ax.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=10)
    ax.imshow(image, aspect='equal')
    current_alpha = alphas[t]
    print(current_alpha)
    val,max_alpha = torch.max(current_alpha,0)
    bbox = test_bb[max_alpha]
    ax.add_patch(
        plt.Rectangle((bbox[0], bbox[1]),
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1], fill=False,
                        edgecolor='red', linewidth=2.0, alpha=0.8)
        )
    # ax.text(bbox[0], bbox[1] - 2,
    #             '{:s} {:.3f}'.format(words[t],int(val)),
    #             bbox=dict(facecolor='blue', alpha=0.5),
    #             fontsize=14, color='white')
    # if smooth:
    #     alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
    # else:
    #     alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
    # if t == 0:
    #     plt.imshow(alpha, alpha=0)
    # else:
    #     plt.imshow(alpha, alpha=0.8)
    # plt.set_cmap(cm.Greys_r)
    # plt.axis('off')
    plt.axis('off')
plt.tight_layout()
plt.draw()
plt.show()
