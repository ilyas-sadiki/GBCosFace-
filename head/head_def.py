import sys
import yaml
sys.path.append('../../')
from head.BaseHead import BaseHead
from head.SubCenterHead import SubCenterHead


class HeadFactory:
    """Factory to produce head according to the head_conf.yaml
    
    Attributes:
        head_type(str): which head will be produce.
        head_param(dict): parsed params and it's value.
    """
    def __init__(self, head_type, head_conf_file, **args):
        self.head_type = head_type
        with open(head_conf_file) as f:
            head_conf = yaml.load(f, Loader=yaml.FullLoader)
            self.head_param = head_conf[head_type]
        if 'num_class' in args:
            self.head_param['num_class'] = args['num_class']
        print('head param:')
        print(self.head_param)
    def get_head(self):
        if self.head_type == 'BaseHead':
            feat_dim = self.head_param['feat_dim'] # dimension of the output features, e.g. 512 
            num_class = self.head_param['num_class'] # number of classes in the training set.
            head = BaseHead(feat_dim, num_class)
        if self.head_type == 'SubCenterHead':
            feat_dim = self.head_param['feat_dim']
            num_class = self.head_param['num_class']
            k = self.head_param.get('k', 3)
            head = SubCenterHead(feat_dim, num_class, k)
        else:
            pass
        return head
