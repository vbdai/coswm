import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from astropy.timeseries import LombScargle
from models import ResNet18
from collections import OrderedDict
import matplotlib.pyplot as plt


def softmax_signal(output, image, k, epsilon, num_classes=10, shape='cosine', linear=False, lin_map=None, padding=0.0):
    N = num_classes
    if linear:
        imageplus = image.view(image.shape[0], image.shape[1] * image.shape[2] * image.shape[3])
        if lin_map is None:
            if image.device.type == 'cpu':
                linmap = torch.ones(imageplus.shape[1], 1)
            else:
                linmap = torch.ones(imageplus.shape[1], 1).cuda()
        else:
            linmap = lin_map

        x = torch.matmul(imageplus, linmap)
        x = x.view((x.shape[0], 1))
    else:
        x = torch.norm(image.view([image.shape[0], image.shape[1] * image.shape[2] * image.shape[3]]),
                       dim=1, keepdim=True)

    if shape == 'cosine':
        phases = torch.tensor([np.pi for i in range(N)])
        phases[0] = 0
        if image.device.type == 'cuda':
            phases = phases.cuda()
        phi = torch.cos(k * x + phases)
    elif shape == 'sawtooth':
        coeffs = torch.tensor([(-1)**i for i in range(N)])

        if N % 2 == 1:
            # If number of classes is odd, rescale even entries so that the total sums to zero
            coeffs[0::2] = coeffs[0::2]*(coeffs[0::2].shape[0]-1)/coeffs[0::2].shape[0]
        if image.device.type == 'cuda':
            coeffs = coeffs.cuda()

        phi = coeffs * (2 * ((k * x) / 2 / np.pi - torch.ceil((k * x) / 2 / np.pi)) + 1)
    elif shape == 'triangle':
        coeffs = torch.tensor([(-1.0) ** i for i in range(N)])
        if N % 2 == 1:
            # If number of classes is odd, rescale even entries so that the total sums to zero
            coeffs[0::2] = coeffs[0::2] * (coeffs[0::2].shape[0] - 1) / coeffs[0::2].shape[0]
        if image.device.type == 'cuda':
            coeffs = coeffs.cuda()
        phi = coeffs * (-2 / np.pi * torch.acos(torch.cos(k * x)) + 1)
    elif shape == 'square':
        coeffs = torch.tensor([(-1.0) ** i for i in range(N)])
        if N % 2 == 1:
            # If number of classes is odd, rescale even entries so that the total sums to zero
            coeffs[0::2] = coeffs[0::2] * (coeffs[0::2].shape[0] - 1) / coeffs[0::2].shape[0]
        if image.device.type == 'cuda':
            coeffs = coeffs.cuda()
        phi = coeffs * (torch.sign(torch.cos(k * x)) + 1)
    else:
        print('softmax_signal -- Invalid shape: returning zero signal')
        phi = 0

    epsilons = torch.tensor([epsilon/(N-1) for i in range(N)])
    epsilons[0] = epsilon

    if image.device.type == 'cpu':
        epsilons = epsilons.cpu()
    else:
        epsilons = epsilons.cuda()

    sm = F.softmax(output, dim=1)
    smsigned = (sm + padding + 1e-25 + epsilons * (1 + phi)) / (1 + (torch.sum(epsilons) + padding + 1e-25))

    return smsigned

def build_periodogram(xy_array, n_freqs=200000, k=0.5):
    '''
    :param xy_array:
    :param labels:
    :param N:
    :param n_freqs:
    :return:

    The returned array has the frequencies as its first column, and the remaining blocks
    of N columns represent the softmax scores of the inputs of a given label i, for the logit
    corresponding to index j
    '''
    freqs_array = np.zeros((n_freqs, 2))
    freqs = np.linspace(0.002, 40, n_freqs)
    freqs_array[:, 0] = freqs

    x = xy_array[:, 0]
    if x.shape[0] == 0:
        freqs_array[:, 1] = np.zeros(n_freqs)
        thetas = [0.0, 0.0, 0.0]
    else:
        y = xy_array[:, 1] - np.mean(xy_array[:, 1])

        ls = LombScargle(x, y, normalization='psd')
        power = ls.power(freqs)

        k_freq = freqs[np.argmin(abs(freqs - k / 2 / np.pi))]
        thetas = ls.model_parameters(k_freq)

        freqs_array[:, 1] = power

    return freqs_array, thetas


def get_spectrum_window(freqs, powers, k, halfwidth=0.001, avg=True):
    idx = (freqs > (k - halfwidth) / 2 / np.pi) & (freqs < (k + halfwidth) / 2 / np.pi)
    not_idx = (freqs <= (k - halfwidth) / 2 / np.pi) | (freqs >= (k + halfwidth) / 2 / np.pi)

    if avg:
        if np.average(powers[not_idx]) == 0.0:
            return 0.0, 0.0
        else:
            return np.average(powers[idx]), np.average(powers[idx]) / np.average(powers[not_idx])
    else:
        return np.sum(powers[idx]), np.average(powers[idx]) * len(idx) / np.average(powers[not_idx])


def load_model(path, filename, num_classes=10, device='cpu', dataset='cifar10', model_arc='resnet18', old=0, offset=13):

    def check_dict(state_dict):
        have_prefix = True
        new_state_dict = OrderedDict()

        for key, value in state_dict.items():
            if key[:13] == 'model.module.':
                name = key[13:]
                new_state_dict[name] = value
            else:
                have_prefix = False

        if have_prefix:
            print("have prefix")
            return new_state_dict
        else:
            return state_dict

    # Obtain correct number of channels in the input layer
    if dataset in ['mnist', 'emnist', 'fmnist']:
        num_channels = 1
    else:
        num_channels = 3

    model_out = ResNet18(num_classes=num_classes, num_channels=num_channels)

    state_dict = torch.load(path + filename)
    state_dict = check_dict(state_dict)

    model_out.load_state_dict(state_dict)

    return model_out.to(device)


def select_split_classes(train_set, test_set, classes, train_part, dataset='cifar10', full_test=True):
    """
    Modifies train and test sets to selected classes and splits train set in two equal parts per class
    :param train_set: training set
    :param test_set: testing set
    :param classes: list of classes to be kept
    :param train_part: Selects what half of the training set to return
                       (must be a string '0' for teacher half and '1' for student half)
    :param dataset: Name of the dataset ('cifar10', 'fmnist')
    :param full_test: Whether to return the full testing set (usually yes, so left True)
    :return:
    """
    idx_total_train = np.zeros(len(train_set.targets), dtype=bool)
    idx_total_test = np.zeros(len(test_set.targets), dtype=bool)
    for c in classes:
        idx = np.array(train_set.targets) == c
        idx_test = np.array(test_set.targets) == c
        if train_part == 0:
            idx_train = idx & (np.cumsum(idx) <= np.sum(idx) // 2)
        elif train_part == 1:
            idx_train = idx & (np.cumsum(idx) > np.sum(idx) // 2)
        else:
            idx_train = idx

        idx_total_train |= idx_train
        idx_total_test |= idx_test

    if dataset == 'cifar10':
        train_set.targets = np.array(train_set.targets)[torch.tensor(idx_total_train)]
        train_set.data = np.array(train_set.data)[torch.tensor(idx_total_train)]
        if not full_test:
            test_set.targets = np.array(test_set.targets)[torch.tensor(idx_total_test)]
            test_set.data = np.array(test_set.data)[torch.tensor(idx_total_test)]

    elif dataset in ['mnist', 'fmnist']:
        train_set.targets = train_set.targets[idx_total_train]
        train_set.data = train_set.data[idx_total_train]
        if not full_test:
            test_set.targets = test_set.targets[idx_total_test]
            test_set.data = test_set.data[idx_total_test]


class CESinPert(nn.Module):
    """
    Modified cross-entropy loss using modified softmax in the logarithm instead of the regular softmax
    """
    def __init__(self, inputs, N=10, k=5.0, epsilon=0.0, shape='cosine', linear=False, lin_map=None, padding=0.0):
        super(CESinPert, self).__init__()
        self.k = k
        self.epsilon = epsilon
        self.inputs = inputs
        self.N = N
        self.shape = shape
        self.linear = linear
        self.lin_map = lin_map
        self.padding = padding


    def forward(self, outputs, labels):
        pertsm = softmax_signal(outputs, self.inputs, self.k, self.epsilon, self.N, self.shape,
                                linear=self.linear, lin_map=self.lin_map, padding=self.padding)

        logpsm = torch.log(pertsm).clamp(min=-50)

        return nn.functional.nll_loss(logpsm, labels)


class CEProbs(nn.Module):
    """
    Custom loss function performing KL loss on soft labels
    """
    def __init__(self, num_classes=10):
        super(CEProbs, self).__init__()
        self.num_classes = num_classes

    def forward(self, predicted, target):
        num_points = predicted.shape[0]
        num_classes = self.num_classes  # Remove unless binary classification
        cum_losses = predicted.new_zeros(num_points)

        for y in range(num_classes):
            target_temp = predicted.new_full((num_points,), y, dtype=torch.long)
            y_loss = F.cross_entropy(predicted, target_temp, reduction="none")
            cum_losses += target[:, y].float() * y_loss

        return cum_losses.mean()
