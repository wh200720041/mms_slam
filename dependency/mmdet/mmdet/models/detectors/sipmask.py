from ..registry import DETECTORS
from .single_stage import SingleStageDetector

from mmdet.core import bbox2result


@DETECTORS.register_module
class SipMask(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SipMask, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained)

    def simple_test(self, img, img_metas, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_metas, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        # bbox_results = [
        #     bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
        #     for det_bboxes, det_labels in bbox_list
        # ]
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels, aa in bbox_list
        ]
        segm_results = [
            aa
            for det_bboxes, det_labels, aa in bbox_list
        ]
        # aa= bbox_list[0][0][:,-1]>0.5
        return bbox_results[0], segm_results[0]
