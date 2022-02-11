import os
import json
import cv2 as cv #3
import PIL.Image
import numpy as np
from labelme import utils #1

import io
import PIL.ImageDraw#2
import os.path as osp
import warnings
import yaml

#moi nguoi chi can thay cai dong thu 107 bang duong dan toi file json cua minh nha

def label_colormap(N=256):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((N, 3))
    for i in range(0, N):
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = (id >> 3)
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    cmap = cmap.astype(np.float32) / 255
    return cmap
# similar function as skimage.color.label2rgb
def label2rgb(lbl, img=None, n_labels=None, alpha=0.5, thresh_suppress=0):
    if n_labels is None:
        n_labels = len(np.unique(lbl))
    cmap = label_colormap(n_labels)
    cmap = (cmap * 255).astype(np.uint8)

    lbl_viz = cmap[lbl]
    lbl_viz[lbl == -1] = (0, 0, 0)  # unlabeled
    # change to Binary image
    for i in range(len(lbl_viz)):
        for j in range(len(lbl_viz[i])):
            if lbl_viz[i][j][0]!=0:
                lbl_viz[i][j]=[255,255,255]
    return lbl_viz
def draw_label(label, img=None, label_names=None, colormap=None):
    import matplotlib.pyplot as plt
    backend_org = plt.rcParams['backend']
    plt.switch_backend('agg')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0,
                        wspace=0, hspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    if label_names is None:
        label_names = [str(l) for l in range(label.max() + 1)]

    if colormap is None:
        colormap = label_colormap(len(label_names))

    label_viz = label2rgb(label, img, n_labels=len(label_names))
    plt.imshow(label_viz)
    plt.axis('off')

    plt_handlers = []
    plt_titles = []
    for label_value, label_name in enumerate(label_names):
        if label_value not in label:
            continue
        if label_name.startswith('_'):
            continue
        fc = colormap[label_value]
        p = plt.Rectangle((0, 0), 1, 1, fc=fc)
        plt_handlers.append(p)
        #code for include title for annotating image
        #start code
    #     plt_titles.append('{name}'
    #                       .format(value=label_value, name=label_name))
    # plt.legend(plt_handlers, plt_titles, loc='lower right', framealpha=.5)
    #end code
    f = io.BytesIO()
    plt.savefig(f, bbox_inches='tight', pad_inches=0)
    plt.cla()
    plt.close()

    plt.switch_backend(backend_org)

    out_size = (label_viz.shape[1], label_viz.shape[0])
    out = PIL.Image.open(f).resize(out_size, PIL.Image.BILINEAR).convert('RGB')
    out = np.asarray(out)
    return out




warnings.warn("This script is aimed to demonstrate how to convert the\n"
                  "JSON file to a single image dataset, and not to handle\n"
                  "multiple JSON files to generate a real-use dataset.")
json_file='/home/gumiho/project/coral_reef/jsonfile'   #json path
list_path = os.listdir(json_file)
for i in range(0, len(list_path)):
    path = os.path.join(json_file, list_path[i])
    if os.path.isfile(path):

        data = json.load(open(path))
        img = utils.img_b64_to_arr(data['imageData'])
        lbl, lbl_names = utils.labelme_shapes_to_label(img.shape, data['shapes'])

        captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]

        lbl_viz = draw_label(lbl, img, captions)

        out_dir = osp.basename(path).split('.json')[0]
        save_file_name = out_dir
        # cv.imshow('img',lbl_viz)
        # cv.waitKey(0)
        img=np.copy(lbl_viz)
        for i in range(len(img)):
            for j in range(len(img[i])):
                if img[i][j][0] != 0:
                    img[i][j] = [255, 255, 255]

        if not osp.exists(json_file + 'mask'):
            os.mkdir(json_file + 'mask')
        maskdir = json_file + 'mask'

        if not osp.exists(json_file + 'mask_converted'):
            os.mkdir(json_file + 'mask_converted')
        maskvizdir = json_file + 'mask_converted'

        out_dir1 = maskdir

        PIL.Image.fromarray(lbl).save(out_dir1 + '/' + save_file_name + '.png')

        PIL.Image.fromarray(img).save(maskvizdir + '/' + save_file_name +
                                          '_label_viz.png')

        with open(osp.join(out_dir1, 'label_names.txt'), 'w') as f:
            for lbl_name in lbl_names:
                f.write(lbl_name + '\n')

        warnings.warn('info.yaml is being replaced by label_names.txt')
        info = dict(label_names=lbl_names)
        with open(osp.join(out_dir1, 'info.yaml'), 'w') as f:
            yaml.safe_dump(info, f, default_flow_style=False)

        print('Saved to: %s' % out_dir1)