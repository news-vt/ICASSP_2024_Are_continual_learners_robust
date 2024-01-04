# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from models.utils.continual_model import ContinualModel
from models.utils.Dist import Nonparametric, get_grad_norm, Normal
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


# class Er(ContinualModel):
#     NAME = 'er'
#     COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

#     def __init__(self, backbone, loss, args, transform):
#         super(Er, self).__init__(backbone, loss, args, transform)
#         self.buffer = Buffer(self.args.buffer_size, self.device)

#     def observe(self, inputs, labels, not_aug_inputs): #(input, labels) are in a batch form

#         real_batch_size = inputs.shape[0]

#         self.opt.zero_grad()
#         # if not self.buffer.is_empty():
#         #     buf_inputs, buf_labels = self.buffer.get_data(
#         #         self.args.minibatch_size, transform=self.transform) #How can I extend it to adding multiple batches? just do for loop
#         #     inputs = torch.cat((inputs, buf_inputs))
#         #     labels = torch.cat((labels, buf_labels))
#         # correct,  total = 0.0, 0.0
#         outputs_stream = self.net(inputs)
#         # _, pred = torch.max(outputs_stream.data, 1)
#         # correct += torch.sum(pred == labels).item()
#         # total += labels.shape[0]        
#         # loss_stream_acc = correct / total * 100

#         loss= self.loss(outputs_stream, labels)

#         if not self.buffer.is_empty():
#             buf_inputs, buf_labels = self.buffer.get_data(
#                 self.args.minibatch_size, transform=self.transform) #How can I extend it to adding multiple batches? just do for loop
#             buf_outputs = self.net(buf_inputs)
#             loss_buf = self.loss(buf_outputs, buf_labels)
#             loss += loss_buf

#         loss.backward()
#         self.opt.step()

#         self.buffer.add_data(examples=not_aug_inputs,
#                              labels=labels[:real_batch_size])

#         return loss.item()
    
#     def memorization(self):
#         status = self.net.training
#         self.net.eval()
#         acc = 0
#         correct, total = 0.0, 0.0

#         if not self.buffer.is_empty():
#             buf_inputs, buf_labels = self.buffer.get_all_data()
#             with torch.no_grad():
#                 outputs = self.net(buf_inputs)
#                 _, pred = torch.max(outputs.data, 1)
#                 correct += torch.sum(pred ==buf_labels).item()
#                 total += buf_labels.shape[0]

#                 acc = correct / total * 100

#         self.net.train(status)

#         return round(acc,2 )

class EQRM(ContinualModel):
    """
    Empirical quantile risk minimization.
    """
    NAME = 'eqrm'
    COMPATIBILITY =  ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform,):
        super(EQRM, self).__init__(backbone, loss, args, transform)
        # self.dist = Nonparametric()
        self.dist = Normal()
        self.grad_ratio = None
        self.update_count = 0
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.erm_grad_norm = -1

    # def update(self, minibatches, unlabeled=None):
    def observe(self, inputs, labels, not_aug_inputs):
        # ERM pretraining def observe(self, inputs, labels, not_aug_inputs):
        real_batch_size = inputs.shape[0]

        if self.update_count < self.args.erm_pretrain_iters:
            return self.ERM_update(inputs, labels)

        # Reset Adam as it doesn't like a sharp jump in gradients when changing objectives
        # if self.update_count == self.args.erm_pretrain_iters:
        #     lr_ = self.args.lr
        #     if self.args.erm_pretrain_iters > 0:
        #         lr_ /= self.args.lr_factor_reduction
        #     self.opt = torch.optim.Adam(
        #         self.net.parameters(),
        #         lr=lr_,
        #         weight_decay=self.args.optim_wd)
            
        if not self.buffer.is_empty(): ##get risks from the memory
            for foo in range(self.args.env_batch):
                buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=None) #How can I extend it to adding multiple batches? just do for loop
                inputs = torch.cat((inputs, buf_inputs))
                labels = torch.cat((labels, buf_labels))
                


        all_x = torch.cat([x for x in inputs])
        # print(all_x[0].size())
        all_y = torch.cat([y.reshape(1) for y in labels]) #zero dimensional verctor causes an error with cat function

        # QRM objective
        env_risks = torch.cat([self.loss(self.net(x.reshape([1,28,28])), y.reshape(1)).reshape(1) for x, y in zip(all_x, all_y)])
          #IDK but, without reshape it gives [28,28] tensors, which are not compatiable with model
        self.dist.estimate_parameters(env_risks)
        alpha = torch.FloatTensor([self.args.eqrm]).to(self.device)
        loss_eqrm = self.dist.icdf(alpha)

        # Rescale gradients if using erm init/pretraining
        if self.args.erm_pretrain_iters > 0:
            if self.grad_ratio is None:
                self.opt.zero_grad()
                loss_eqrm.backward(retain_graph=True)
                self.grad_ratio = get_grad_norm(self.net) / self.erm_grad_norm
            loss_eqrm = loss_eqrm / self.grad_ratio

        #Batch from the buffer
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=None) 
            buf_outputs = self.net(buf_inputs)
            loss_buf = self.loss(buf_outputs, buf_labels)
            loss_eqrm = loss_eqrm * self.args.balance + loss_buf 

        # Step
        self.opt.zero_grad()
        loss_eqrm.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                        labels=labels[:real_batch_size])

        self.update_count += 1
        return loss_eqrm.item()
    
    def ERM_update(self, inputs, labels, unlabeled=None):
        # all_x = torch.cat([x for x, y in minibatches])
        # all_y = torch.cat([y for x, y in minibatches])
        output = self.net(inputs)
        loss = self.loss(output, labels)

        self.opt.zero_grad()
        loss.backward()
        if self.erm_grad_norm < 0:              # store initial grad norm for normalizing regularization losses
            self.erm_grad_norm *= -get_grad_norm(self.net)
        self.opt.step()

        self.update_count += 1
        return loss.item()
    
    def memorization(self):
        status = self.net.training
        self.net.eval()
        acc = 0
        correct, total = 0.0, 0.0

        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_all_data()
            with torch.no_grad():
                outputs = self.net(buf_inputs)
                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred ==buf_labels).item()
                total += buf_labels.shape[0]

                acc = correct / total * 100

        self.net.train(status)

        return round(acc,2 )
    






# class Algorithm(nn.Module):
#     """
#     A subclass of Algorithm implements a domain generalization algorithm.
#     Subclasses should implement the following:
#     - update()
#     - predict()
#     """

#     def __init__(self, network, hparams, loss_fn):
#         super(Algorithm, self).__init__()
#         self.network = network
#         self.hparams = hparams
#         self.loss_fn = loss_fn
#         self.register_buffer('update_count', torch.tensor([0]))

#     def update(self, minibatches, unlabeled=None):
#         """
#         Perform one update step, given a list of (x, y) tuples for all
#         environments.

#         Admits an optional list of unlabeled minibatches from the test domains,
#         when task is domain_adaptation.
#         """
#         raise NotImplementedError

#     def predict(self, x):
#         raise NotImplementedError  

# class ERM(Algorithm):
#     """
#     Empirical Risk Minimization (ERM)
#     """

#     def __init__(self, network, loss_fn, hparams):
#         super(ERM, self).__init__(network, loss_fn, hparams)
#         self.optimizer = torch.optim.AdamW(
#             self.network.parameters(),
#             lr=self.hparams["lr"],
#             weight_decay=self.hparams['weight_decay']
#         )
#         self.register_buffer("erm_grad_norm", torch.tensor([-1.]))

#     def update(self, minibatches, unlabeled=None):
#         all_x = torch.cat([x for x, y in minibatches])
#         all_y = torch.cat([y for x, y in minibatches])
#         loss = self.loss_fn(self.predict(all_x), all_y)

#         self.optimizer.zero_grad()
#         loss.backward()
#         if self.erm_grad_norm < 0:              # store initial grad norm for normalizing regularization losses
#             self.erm_grad_norm *= -get_grad_norm(self.network)
#         self.optimizer.step()

#         self.update_count += 1
#         return {'loss': loss.item()}

#     def predict(self, x):
#         return self.network(x)