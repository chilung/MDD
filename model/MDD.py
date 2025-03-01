import torch.nn as nn
import model.backbone as backbone
import torch.nn.functional as F
import torch
import numpy as np

class GradientReverseLayer(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input_, backward_tensor):
        ctx.save_for_backward(input_, backward_tensor)
        output_ = input_ * 1.0
        return output_

    @staticmethod
    def backward(ctx, grad_output):
        _, backward_tensor = ctx.saved_tensors
        iter_num = backward_tensor[0]
        alpha = backward_tensor[1]
        low_value = backward_tensor[2]
        high_value = backward_tensor[3]
        max_iter = backward_tensor[4]
        
        coeff = np.float(
            2.0 * (high_value - low_value) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (
                        high_value - low_value) + low_value)
        # print('coeff = {}, iter_num = {}, alpha = {}, low_value = {}, high_value = {}, max_iter = {}'.format(coeff, iter_num, alpha, low_value, high_value, max_iter))
        
        return -coeff * grad_output, None

class GradientReverseLayer_ver1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

# def grad_reverse(x):
#     return GradReverse.apply(x)

# This is the original version of GRL
class GradientReverseLayer_org(torch.autograd.Function):
    def __init__(self, iter_num=0, alpha=1.0, low_value=0.0, high_value=0.1, max_iter=1000.0):
        self.iter_num = iter_num
        self.alpha = alpha
        self.low_value = low_value
        self.high_value = high_value
        self.max_iter = max_iter

    def forward(self, input):
        self.iter_num += 1
        output = input * 1.0
        return output

    def backward(self, grad_output):
        self.coeff = np.float(
            2.0 * (self.high_value - self.low_value) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter)) - (
                        self.high_value - self.low_value) + self.low_value)
        return -self.coeff * grad_output


class MDDNet(nn.Module):
    def __init__(self, base_net='ResNet50', use_bottleneck=True, bottleneck_dim=1024, width=1024, class_num=31, 
                 iter_num=0, alpha=1.0, low_value=0.0, high_value=0.1, max_iter=1000.0):

        super(MDDNet, self).__init__()
        ## set base network
        self.base_network = backbone.network_dict[base_net]()
        self.use_bottleneck = use_bottleneck
        self.grl_layer = GradientReverseLayer()
        self.bottleneck_layer_list = [nn.Linear(self.base_network.output_num(), bottleneck_dim), nn.BatchNorm1d(bottleneck_dim), nn.ReLU(), nn.Dropout(0.5)]
        self.bottleneck_layer = nn.Sequential(*self.bottleneck_layer_list)
        self.classifier_layer_list = [nn.Linear(bottleneck_dim, width), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(width, class_num)]
        self.classifier_layer = nn.Sequential(*self.classifier_layer_list)
        self.classifier_layer_2_list = [nn.Linear(bottleneck_dim, width), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(width, class_num)]
        self.classifier_layer_2 = nn.Sequential(*self.classifier_layer_2_list)
        self.softmax = nn.Softmax(dim=1)

        ## initialization
        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)
        for dep in range(2):
            self.classifier_layer_2[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer_2[dep * 3].bias.data.fill_(0.0)
            self.classifier_layer[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer[dep * 3].bias.data.fill_(0.0)


        ## collect parameters
        self.parameter_list = [{"params":self.base_network.parameters(), "lr":0.1},
                            {"params":self.bottleneck_layer.parameters(), "lr":1},
                        {"params":self.classifier_layer.parameters(), "lr":1},
                               {"params":self.classifier_layer_2.parameters(), "lr":1}]

        ## for backward tensor
        self.backward_tensor = torch.tensor([iter_num, alpha, low_value, high_value, max_iter])

    def forward(self, inputs):
        features = self.base_network(inputs)
        if self.use_bottleneck:
            features = self.bottleneck_layer(features)
        # features_adv = self.grl_layer(features)
        
        self.backward_tensor[0] += 1  # iter_num
        features_adv = self.grl_layer.apply(features, self.backward_tensor)
        outputs_adv = self.classifier_layer_2(features_adv)
        
        outputs = self.classifier_layer(features)
        softmax_outputs = self.softmax(outputs)

        print('===forward===')
        print('features dim: {}'.format(features.size(0)))
        print('outputs dim: {}'.format(outputs.size(0)))
        print('softmax_outputs dim: {}'.format(softmax_outputs.size(0)))
        print('outputs_adv dim: {}'.format(outputs_adv.size(0)))
        print('===forward===')
        return features, outputs, softmax_outputs, outputs_adv

class MDD(object):
    def __init__(self, base_net='ResNet50', width=1024, class_num=31, use_bottleneck=True, use_gpu=True, srcweight=3):
        self.c_net = MDDNet(base_net, use_bottleneck, width, width, class_num)

        self.use_gpu = use_gpu
        self.is_train = False
        self.iter_num = 0
        self.class_num = class_num
        if self.use_gpu:
            self.c_net = self.c_net.cuda()
        self.srcweight = srcweight


    def get_loss(self, inputs, labels_source):
        class_criterion = nn.CrossEntropyLoss()

        _, outputs, _, outputs_adv = self.c_net(inputs)

        print('inputs dim: {}'.format(inputs.size(0)))
        print('outputs dim: {}'.format(outputs.size(0)))
        print('outputs_adv dim: {}'.format(outputs_adv.size(0)))
        print('labels_source dim: {}'.format(labels_source.size(0)))
        print('labels_source dim: {} labels_source: {}'.format(labels_source.size(0), labels_source))
        classifier_loss = class_criterion(outputs.narrow(0, 0, labels_source.size(0)), labels_source)

        target_adv = outputs.max(1)[1]
        target_adv_src = target_adv.narrow(0, 0, labels_source.size(0))
        target_adv_tgt = target_adv.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0))

        classifier_loss_adv_src = class_criterion(outputs_adv.narrow(0, 0, labels_source.size(0)), target_adv_src)

        # logloss_tgt = torch.log(
        #     1 - F.softmax(outputs_adv.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0)), dim = 1))
        logloss_tgt = torch.log(torch.clamp(
            1 - F.softmax(outputs_adv.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0)), dim = 1), min=1e-15))
        
        classifier_loss_adv_tgt = F.nll_loss(logloss_tgt, target_adv_tgt)

        print('classifier_loss_adv_src: {}, classifier_loss_adv_tgt: {}'.format(classifier_loss_adv_src, classifier_loss_adv_tgt))
        transfer_loss = self.srcweight * classifier_loss_adv_src + classifier_loss_adv_tgt

        self.iter_num += 1
        total_loss = classifier_loss + transfer_loss

        return total_loss

    def predict(self, inputs):
        _, _, softmax_outputs,_= self.c_net(inputs)
        return softmax_outputs

    def get_parameter_list(self):
        return self.c_net.parameter_list

    def set_train(self, mode):
        self.c_net.train(mode)
        self.is_train = mode
