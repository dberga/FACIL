import torch
from torch import nn
from copy import deepcopy
from utils import visualize_batch


class LLL_Net(nn.Module):
    """ Basic class for implementing networks """

    def __init__(self, model, remove_existing_head=False):
        head_var = model.head_var
        assert type(head_var) == str
        assert not remove_existing_head or hasattr(model, head_var), \
            "Given model does not have a variable called {}".format(head_var)
        assert not remove_existing_head or type(getattr(model, head_var)) in [nn.Sequential, nn.Linear], \
            "Given model's head {} does is not an instance of nn.Sequential or nn.Linear".format(head_var)
        super(LLL_Net, self).__init__()

        self.model = model
        last_layer = getattr(self.model, head_var)

        if remove_existing_head:
            if type(last_layer) == nn.Sequential:
                self.out_size = last_layer[-1].in_features
                # strips off last linear layer of classifier
                del last_layer[-1]
            elif type(last_layer) == nn.Linear:
                self.out_size = last_layer.in_features
                # converts last layer into identity
                # setattr(self.model, head_var, nn.Identity())
                # WARNING: this is for when pytorch version is <1.2
                self.model.fc = nn.Sequential()
        else:
            self.out_size = last_layer.out_features

        self.heads = nn.ModuleList()
        self.task_cls = []
        self.task_offset = []

        self._initialize_weights()

    def add_head(self, head_architecture):
        self.heads.append(head_architecture)
        # we re-compute instead of append in case an approach makes changes to the heads
        self.task_cls = torch.tensor([head.out_features for head in self.heads])
        self.task_offset = torch.cat([torch.LongTensor(1).zero_(), self.task_cls.cumsum(0)[:-1]])

    # Simplification to work on multi-head only -- returns all head outputs in a list
    def forward(self, x, return_features=True):
        #gx = self.model.glayer(x)
        #visualize_batch(gx.cpu(),"debug")
        x = self.model(x) #here birds and flowers output is 512x7x7 but in cifar is 512, did we see this bug before?
        if x is tuple: #disentanglement (before "x" and after "xx" flatten operation)
            xx=x[1].clone()
            x=x[0]
        assert (len(self.heads) > 0), "Cannot access any head"
        y = []
        for head in self.heads:
            y.append(head(x))
        if return_features and 'xx' in locals():
            return y, xx
        elif return_features and not 'xx' in locals():
            return y, x
        else:
            return y

    def get_copy(self):
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        self.load_state_dict(deepcopy(state_dict))
        return

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False
        return

    def _initialize_weights(self):
        # TODO: add the different initializations
        pass
