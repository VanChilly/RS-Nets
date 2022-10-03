# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import numpy as np
import torch
import torch.nn as nn


class GumbelSoftmax(nn.Module):
    def __init__(self, hard=True):
        super().__init__()
        self.training = False
        self.hard = hard
        self.gpu = False
        self.minval = 0.0
        self.maxval = 1.0
        # self.updates = torch.tensor(np.array([1.0]), dtype=torch.float32).to(device)

    def cuda(self):
        self.gpu = True

    def cpu(self):
        self.gpu = False

    def sample_gumbel_like(self, template_tensor, eps=1e-10):
        uniform_samples_tensor = torch.randn(template_tensor.shape).to(template_tensor.device)
        uniform_samples_tensor.uniform_(self.minval, self.maxval)

        uniform_samples_tensor = uniform_samples_tensor.abs()
        gumbel_samples_tensor = - (eps - (uniform_samples_tensor + eps).log()).log()
        return gumbel_samples_tensor

    def gumbel_softmax_sample(self, logits, tau):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        # no tau
        gumbel_samples_tensor = self.sample_gumbel_like(logits)
        assert 0 < tau <= 1, f"Tau is not between 0 and 1, tau is {tau}"
        tau = torch.tensor(tau).to(logits.device)
        gumbel_trick_log_prob_samples = (logits + gumbel_samples_tensor) / tau
        soft_samples = nn.Softmax(dim=-1)(gumbel_trick_log_prob_samples)
        return soft_samples

    def gumbel_softmax(self, logits, hard=True, tau=1):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
        logits: [batch_size, n_class] unnormalized log-probs
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
        """
        y = self.gumbel_softmax_sample(logits, tau)
        y_hard = []
        max_value_indexes = []
        if hard:
            max_value_indexes = y.argmax(axis=1)
            y_hard = torch.zeros_like(logits)
            for batch_idx in range(logits.shape[0]):
                # gentrate one hot prediction: [N, Num_Resolution]
                y_hard[batch_idx][max_value_indexes[batch_idx]] = 1.0 
        return y_hard

    def forward(self, logits, temp=1, force_hard=True):
        result = 0
        if self.training and not force_hard:
            result = self.gumbel_softmax(logits, hard=False, tau=temp)
        else:
            result = self.gumbel_softmax(logits, hard=True, tau=temp)
        return result

