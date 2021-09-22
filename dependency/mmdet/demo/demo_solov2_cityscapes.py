from scipy import ndimage
import mmcv
import torch
import cv2
import os

import numpy as np
import matplotlib.cm as cm
from mmdet.apis import inference_detector, init_detector
from mmdet.core import get_classes

from argparse import ArgumentParser
import glob
import time


def show_result_solo_pyplot(model, img, result, score_thr=0.3, fig_size=(15, 10)):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
    """
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result_solo(img, result, model.CLASSES, score_thr=score_thr, show=False)
    plt.figure(figsize=fig_size)
    plt.imshow(mmcv.bgr2rgb(img))
    plt.show()


def vis_seg(img, result, score_thr, save_dir):
    class_names = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
    print(class_names)
    imgs = [img]
    if result[0]:
        for img, cur_result in zip(imgs, result):
            h, w, _ = img.shape
            img_show = img[:h, :w, :]
            
            seg_label = cur_result[0]
            seg_label = seg_label.cpu().numpy().astype(np.uint8)

            cate_label = cur_result[1]
            cate_label = cate_label.cpu().numpy()

            score = cur_result[2].cpu().numpy()

            vis_inds = score > score_thr
            seg_label = seg_label[vis_inds]
            num_mask = seg_label.shape[0]
            cate_label = cate_label[vis_inds]
            cate_score = score[vis_inds]

            mask_density = []
            for idx in range(num_mask):
                cur_mask = seg_label[idx, :, :]
                cur_mask = mmcv.imresize(cur_mask, (w, h))
                cur_mask = (cur_mask > 0.5).astype(np.int32)
                mask_density.append(cur_mask.sum())

            orders = np.argsort(mask_density)
            seg_label = seg_label[orders]
            cate_label = cate_label[orders]
            cate_score = cate_score[orders]

            seg_show = img_show.copy()
            for idx in range(num_mask):
                idx = -(idx+1)
                cur_mask = seg_label[idx, :,:]
                cur_mask = mmcv.imresize(cur_mask, (w, h))
                cur_mask = (cur_mask > 0.5).astype(np.uint8)

                if cur_mask.sum() == 0:
                    continue

                color_mask = np.random.randint(
                    0, 256, (1, 3), dtype=np.uint8)
                cur_mask_bool = cur_mask.astype(np.bool)
                contours, _ = cv2.findContours(cur_mask*255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                seg_show[cur_mask_bool] = img_show[cur_mask_bool] * 0.4 + color_mask * 0.6

                color_mask = color_mask[0].tolist()
                cv2.drawContours(seg_show, contours, -1, tuple(color_mask), 1, lineType=cv2.LINE_AA)

                cur_cate = cate_label[idx]
                cur_score = cate_score[idx]
                label_text = class_names[cur_cate]

                center_y, center_x = ndimage.measurements.center_of_mass(cur_mask)
                vis_pos = (max(int(center_x) - 10, 0), int(center_y))
                cv2.putText(seg_show, label_text, vis_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), lineType=cv2.LINE_AA)
                cv2.putText(seg_show, '{:.1f}%'.format(cur_score*100), (vis_pos[0], vis_pos[1] + 9),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), lineType=cv2.LINE_AA)
        mmcv.imshow(seg_show)
    else:
        print('no detections')


@torch.no_grad()
def main():
    parser = ArgumentParser()
    parser.add_argument('-d', '--data_f', default='demo/demo.jpg', help='Image file')
    parser.add_argument('-c', '--config', default='configs/solov2/solov2_r101_3x.py', help='Config file')
    parser.add_argument('-w', '--checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--save_dir', default='output/save', help='Device used for save')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    
    data_f = args.data_f

    model = init_detector(args.config, args.checkpoint, device=args.device)
    print('model ready.')
    
    if os.path.isdir(data_f):
        img_files = glob.glob(os.path.join(data_f, '*.[jp][pn]g'))
        for img_f in img_files:
            tic = time.time()
            result = inference_detector(model, img_f)
            cost = time.time() - tic
            print('finished in {}, fps: {}'.format(cost, 1/cost))
            score_thr = args.score_thr
            img = mmcv.imread(img_f)
            vis_seg(img, result, score_thr=score_thr, save_dir=args.save_dir)
    elif 'mp4' in data_f:
        pass
    else:
        result = inference_detector(model, data_f)
        score_thr = args.score_thr

        img = mmcv.imread(data_f)
        vis_seg(img, result, score_thr=score_thr, save_dir=args.save_dir)


if __name__ == '__main__':
    main()
