# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class Er(ContinualModel):
    NAME = 'er'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Er, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

    def observe(self, inputs, labels, not_aug_inputs): #(input, labels) are in a batch form

        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        # if not self.buffer.is_empty():
        #     buf_inputs, buf_labels = self.buffer.get_data(
        #         self.args.minibatch_size, transform=self.transform) #How can I extend it to adding multiple batches? just do for loop
        #     inputs = torch.cat((inputs, buf_inputs))
        #     labels = torch.cat((labels, buf_labels))
        correct,  total = 0.0, 0.0
        outputs_stream = self.net(inputs)
        _, pred = torch.max(outputs_stream.data, 1)
        correct += torch.sum(pred == labels).item()
        total += labels.shape[0]        
        loss_stream_acc = correct / total * 100

        loss= self.loss(outputs_stream, labels)

        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform) #How can I extend it to adding multiple batches? just do for loop
            buf_outputs = self.net(buf_inputs)
            loss_buf = self.loss(buf_outputs, buf_labels)
            loss += loss_buf

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])

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