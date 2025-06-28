import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *
from models.teen.helper import *

class MYNET(nn.Module):
    def __init__(self, args, mode=None):
        super().__init__()
        self.mode = mode
        self.args = args
        if self.args.dataset in ['cifar100','manyshotcifar']:
            self.encoder = resnet20()
            self.num_features = 64
        if self.args.dataset in ['mini_imagenet','manyshotmini','imagenet100','imagenet1000', 'mini_imagenet_withpath']:
            #self.encoder = resnet18(False, args)  # pretrained=False
            self.encoder = resnet18(True, args)  # pretrained=False
            self.num_features = 512
        if self.args.dataset in ['cub200','manyshotcub']:
            self.encoder = resnet18(True, args)  # pretrained=True follow TOPIC, models for cub is imagenet pre-trained. https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
            self.num_features = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.embeddings = {}  # Persistent storage
        self._register_hooks()
        self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)
        self.fc1 = nn.Linear(self.num_features, self.args.num_classes, bias=False)

    def forward_metric(self, x):
        x = self.encode(x)
        #x1 = self.fc1(x)
        x1 = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc1.weight, p=2, dim=-1))
        x1 = self.args.temperature * x1
        if 'cos' in self.mode:
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x

        elif 'dot' in self.mode:
            x = self.fc(x)
        return x,x1

    def encode(self, x):
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x

    def forward(self, input):
        if self.mode != 'encoder':
            input, input1 = self.forward_metric(input)
            return input,input1
        elif self.mode == 'encoder':
            input = self.encode(input)
            return input
        else:
            raise ValueError('Unknown mode')

    def update_fc(self,model, dataloader,class_list,session):
        for batch in dataloader:
            data1, label = [_.cpu() for _ in batch] #[_.cuda() for _ in batch]
            data = self.encode(data1).detach()
            # data = innerFeature(model, data, data1, self.args)
            if self.args.not_data_init:
                new_fc = nn.Parameter(
                    torch.rand(len(class_list), self.num_features, device="cuda"),requires_grad=True)
                nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
            else:
                new_fc = self.update_fc_avg(data, label, class_list)

    def update_fc_avg(self,data,label,class_list):
        new_fc=[]
        for class_index in class_list:
            data_index=(label==class_index).nonzero().squeeze(-1)
            embedding=data[data_index]
            proto=embedding.mean(0)
            new_fc.append(proto)
            self.fc.weight.data[class_index]=proto
        new_fc=torch.stack(new_fc,dim=0)
        return new_fc

    def replace_base_fc(self, trainset, transform, model, args):
        # replace fc.weight with the embedding average of train data
        model = model.eval()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128, num_workers=8, pin_memory=True, shuffle=False)
        trainloader.dataset.transform = transform
        embedding_list = []
        label_list = [];
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                data, label = [_.cpu() for _ in batch]  # [_.cuda() for _ in batch]
                model.mode = 'encoder'  # model.module.mode = 'encoder'
                embedding = model(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        proto_list = []
        for class_index in range(args.base_class):
            data_index = (label_list == class_index).nonzero()
            embedding_this = embedding_list[data_index.squeeze(-1)]
            embedding_this = embedding_this.mean(0)
            proto_list.append(embedding_this)
        proto_list = torch.stack(proto_list, dim=0)
        model.fc.weight.data[:args.base_class] = proto_list  # module.
        return model


    def _register_hooks(self):
        """Register forward hooks once before training starts."""
        def hook_fn(module, input, output):
            output = F.adaptive_avg_pool2d(output, (1, 1))
            output = output.view(output.size(0), -1)
            self.embeddings['kk40'] = output
        if self.args.dataset == 'cifar100':
            relu_layer = self.encoder.layer3[0].relu
        else:
            relu_layer = self.encoder.layer4[0].relu
        relu_layer.register_forward_hook(hook_fn)

