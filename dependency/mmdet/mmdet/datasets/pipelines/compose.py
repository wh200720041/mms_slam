import collections
import numpy as np
from mmdet.utils import build_from_cfg
from ..registry import PIPELINES


@PIPELINES.register_module
class Compose(object):

    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        for t in self.transforms:
            #print(data)
            # print("hello", data['mask_fields'])
            # if(data['mask_fields'] == ['gt_masks'] ):
            #     print('show data')
            #     # print(data['img_shape'])
            #     print(type(data['img']))
            #     print(type(data['gt_masks']))
            #     if(isinstance(data['img'], np.ndarray)):
            #         print(data['img'].shape)
                # else:
                #     print(data['img'].size)
                # print(data['gt_masks'][0].shape)
            #this code is only for training
            if 'img' in data and 'mask_fields' in data and data['mask_fields'] == ['gt_masks']:
                if(isinstance(data['img'], np.ndarray) == True and isinstance(data['gt_masks'], np.ndarray) == True):
                    if(data['img'].shape[0]<data['gt_masks'][0].shape[0]):
                        print('discard',data['filename'])
                        return None
            data = t(data)
            if data is None:
                return None
        return data


    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
