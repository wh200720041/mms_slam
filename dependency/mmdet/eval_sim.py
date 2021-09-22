"""


Debug on eval bug

it seems evaluation can not be done on RLE format annotation,
it should force to iscrowd=1 so that iou can be calculated.

"""
from pycocotools.coco import COCO
from tools.cocoeval import COCOeval
import json


ann_type = 'segm'

ann_f = 'data/cityscapes/annotations/instancesonly_filtered_gtFine_val.json'
r_f = 'results_solo.pkl.segm.json'
c_gt = COCO(ann_f)

anns = json.load(open(r_f))
ii = [i['image_id'] for i in anns]
print(ii)
print(c_gt.getImgIds())

c_dt = c_gt.loadRes(r_f)

print(c_gt.getImgIds())
print(c_dt.getImgIds())

print('GT of image 0: ')
print(c_gt.load_anns(c_gt.getAnnIds(0)))

print('\nRes of image 0:')
print(c_dt.load_anns(c_dt.getAnnIds(0)))


coco_eval = COCOeval(c_gt, c_dt, ann_type)
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()


