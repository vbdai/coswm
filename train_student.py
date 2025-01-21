from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np
import time
import argparse

# Custom loss function
from utils import softmax_signal, CEProbs, select_split_classes, load_model
from models import ResNet18


def train(train_set=None, test_set=None, batch_size=64, num_epochs=40, sched='stepLR',
          steps=1, initial_lr=0.01, final_lr=0.001, opt='SGD', weight_decay=5e-4, momentum=0.9, eval_every_n_epochs=200,
          print_freq=50, num_workers=4, classes=list(range(10)),
          k=0.5, epsilon=0.0, padding=0.0, vec_ind=[0], shape='cosine', model_arc='resnet18', dataset='cifar10',
          filename='', teacher_filename='', train_part=1, method='ours',
          root='models/', path='', teacher_path='', root_data='datasets/'):

    # count for the # of steps
    step_count = 0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if dataset == 'cifar10':
        # Sets input channels for the model
        num_channels = 3
        # Loading dataset
        if train_set is None:
            train_transform = transforms.Compose([
                transforms.ToTensor()
            ])
            train_set = torchvision.datasets.CIFAR10(root=root_data + 'cifar10/', download=False,
                                                     transform=train_transform)
            test_transform = transforms.Compose([
                transforms.ToTensor()
            ])
            test_set = torchvision.datasets.CIFAR10(root=root_data + 'cifar10/', download=False,
                                                    train=False, transform=test_transform)
            # See teacher script for explanation of the dataset split (inverted inequalities to get other half of the split)
            select_split_classes(train_set, test_set, classes, train_part, dataset=dataset, full_test=True)

    # Everything else returns FMNIST
    else:
        # Sets input channels for the model
        num_channels = 1
        # Loading dataset
        if train_set is None:
            train_transform = transforms.Compose([
                transforms.ToTensor()
            ])
            train_set = torchvision.datasets.FashionMNIST(root=root_data + 'fmnist/', download=False,
                                                          transform=train_transform)
            test_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            test_set = torchvision.datasets.FashionMNIST(root=root_data + 'fmnist/', download=False,
                                                         train=False, transform=test_transform)
            # Select one half of the training set
            select_split_classes(train_set, test_set, classes, train_part, dataset=dataset, full_test=True)

    # if dataset == 'cifar10':
    #     train_set_id = SplitDataset(train_set.data, train_set.targets, N=2, transforms=train_set.transform)
    # else:
    #     train_set_id = SplitDataset(train_set.data.numpy(), train_set.targets.numpy(), N=2,
    #                                 transforms=train_set.transform)

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers
    )
    train_loader_stable = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        num_workers=num_workers
    )
    val_loader = test_loader

    # Number of classes and phases for phi_j
    num_classes = len(classes)

    if epsilon != 0.0:
        # Load random mapping projection
        filename_map = 'rand_map_%s.csv' % dataset

        lin_map = []
        lin_map_all = torch.tensor(np.loadtxt(filename_map)).to(device).float()
        for i in vec_ind:
            lin_map.append(lin_map_all[:, i].view(lin_map_all.shape[0], 1))

    model = ResNet18(num_classes=num_classes, num_channels=num_channels).to(device)

    # teacher models loading
    teachers = []
    i_fname = 0
    for fname in teacher_filename:
        teacher = load_model(root + teacher_path, fname + '.pth', num_classes=num_classes, model_arc=model_arc,
                             dataset=dataset, old=0, device=device, offset=0)

        teacher.eval()

        teachers.append(teacher)

        i_fname += 1

    pseudolabels = []

    for images, labels in train_loader_stable:
        images = images.to(device)
        labels = labels.to(device)

        ps_batch = torch.zeros([labels.shape[0], num_classes]).to(device)

        for i in range(len(teachers)):
            if (method[i] == 'ours') & (epsilon != 0.0):
                out_temp = softmax_signal(teachers[i](images), images, k=k, epsilon=epsilon,
                                          num_classes=num_classes, linear=True, lin_map=lin_map[i])
            else:
                out_temp = F.softmax(teachers[i](images), dim=1)

            ps_batch += out_temp / len(teachers)

        pseudolabels.append(ps_batch.detach().cpu().numpy())

    train_set.targets = np.concatenate(pseudolabels)

    # define opimizers
    if opt == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=momentum,
                              weight_decay=weight_decay, nesterov=True)
    elif opt == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    else:
        print('no optimizer')

    # define learning rate schedulers
    total_steps = num_epochs * len(train_set) / batch_size
    if sched == 'stepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs / (steps + 1),
                                                    gamma=(final_lr / initial_lr) ** (1.0 / steps))
    elif sched == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=(final_lr / initial_lr) ** (
                    1.0 / total_steps))
    elif sched == 'stepLR_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=np.sqrt(0.1), cooldown=0,
                                                               patience=5, min_lr=final_lr)
    else:
        print('no scheduler')

    print('\n\n************TRAINING STARTED*************')
    print('k: ', k, ', epsilon: ', epsilon)
    print('path: ', path)
    print('gpus: ', torch.cuda.device_count())
    print('batch: ', batch_size)
    print('num_workers: ', num_workers)
    print('scheduler: ', sched, ', learning rate: initial =', initial_lr, ', final =', final_lr)

    best_acc = 0.0

    # training loop
    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0

        print('\n\n>>>>>>>>>>' + 'starting epoch ' + str(epoch + 1) + ' of ' + str(num_epochs) + '<<<<<<<<<<')

        # data is a list of [inputs, labels]
        for images, labels in train_loader:

            step_count += 1
            model.train()
            # zero parameter gradients
            optimizer.zero_grad()

            # perform calculation on GPU
            labels = labels.to(device)
            images = images.to(device)

            # evaluate with model
            output = model(images)

            # calculate loss
            criterion = CEProbs(num_classes=num_classes)
            train_loss = criterion(output, labels)

            # get accuracy
            total_train = len(labels)
            _, model_prediction = torch.max(output.data, 1)
            correct_train = (model_prediction == torch.argmax(labels, dim=1)).sum().item()
            train_acc = correct_train / total_train

            epoch_loss += train_loss.data

            # back propogation + optimize
            train_loss.backward()
            optimizer.step()

            if sched == 'stepLR_plateau':
                lr = optimizer.param_groups[0]['lr']
            else:
                lr = scheduler.get_lr()[0]

            # print statistics to terminal based on print_freq
            if (step_count % print_freq == 0):
                print('step: ' + str(step_count) + ' | loss: ' + str(round(train_loss.item(), 4)) + ' | acc: ' + str(
                    train_acc) + ' | lr: ' + "{:.3e}".format(lr))

        # perform validation based on eval_every_n_epochs
        if (epoch % eval_every_n_epochs == 0):
            start_time = time.time()
            correct = 0
            total = 0
            model.eval()
            with torch.no_grad():
                for images, labels in val_loader:
                    labels = labels.to(device)
                    images = images.to(device)

                    pslabels = torch.zeros([images.shape[0], num_classes]).to(device)
                    for i in range(len(teachers)):
                        pslabels_raw = teachers[i](images)
                        if method[i] == 'ours':
                            pslabels += softmax_signal(pslabels_raw, images, k, epsilon, num_classes,
                                                       shape=shape, linear=True, lin_map=lin_map[i],
                                                       padding=padding)
                        else:
                            pslabels += F.softmax(pslabels_raw, dim=1)
                    pslabels = pslabels / len(teachers)

                    outputs = model(images)

                    _, model_validation = torch.max(outputs.data, 1)
                    criterion = CEProbs(num_classes=num_classes)
                    val_loss = criterion(outputs, pslabels)
                    total += labels.size(0)
                    correct += (model_validation == labels).sum().item()

                val_acc = round(correct / total, 5)

                is_best = val_acc > best_acc
                best_acc = max(val_acc, best_acc)

                if is_best:
                    # Saving the model
                    print('saving model...')
                    if k % 1 == 0:  # If k is a int
                        k_str = str(int(k))
                    else:
                        k_str = str(k).replace('.', 'p')
                    path_save = root + path
                    if filename == '':
                        fname = 'eps' + str(epsilon).split('.')[1] + 'k' + k_str + '_teacher_binary_split.pth'
                    else:
                        fname = filename + '.pth'
                    torch.save(model.state_dict(), path_save + fname)

            print('---- validation took ', round(time.time() - start_time, 1), ' secs ', ' | val_acc: ', val_acc)

        # adjust learning rate according to lr_scheduler
        if sched == 'stepLR_plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # epoch statistics
        print('epoch took ', round(time.time() - epoch_start, 1), ' seconds')
    print('>>>>>>>>>>>>>>>>training finished<<<<<<<<<<<<<<<<<<<')

    # Testing
    print('*****************testing******************')

    test_loss = 0
    class_correct = list(0 for i in classes)
    class_total = list(0 for i in classes)

    # Load best model
    model = load_model(path_save, fname, num_classes=num_classes, device=device, dataset=dataset, model_arc=model_arc)

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            # CUDA verification for labels
            if torch.cuda.is_available():
                labels = labels.cuda()
                images = images.cuda()
            # compute predictions score

            pslabels = torch.zeros([images.shape[0], num_classes]).to(device)
            for i in range(len(teachers)):
                pslabels_raw = teachers[i](images)
                if method[i] == 'ours':
                    pslabels += softmax_signal(pslabels_raw, images, k, epsilon, num_classes,
                                               shape=shape, linear=True, lin_map=lin_map[i],
                                               padding=padding)
                else:
                    pslabels += F.softmax(pslabels_raw, dim=1)
            pslabels = pslabels / len(teachers)

            outputs = model(images)

            # Compute the loss on the predictions and update the test loss
            criterion = CEProbs(num_classes=num_classes)
            loss = criterion(outputs, pslabels)
            test_loss += loss.item()*images.size(0)
            # Class predictions to the highest score
            _, predictions = torch.max(outputs, 1)
            # Compare with ground truth
            correct = (predictions == labels).squeeze()
            # Test accuracy computation
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

    # Normalize test loss
    test_loss = test_loss/len(test_loader.dataset)
    print('Test loss: {:.6f}\n'.format(test_loss))

    for i in range(num_classes):
        if class_total[i] > 0:
            print('Test accuracy of %5s: %2.2f%% (%2d/%2d)'%
                  (str(i), 100 * class_correct[i]/class_total[i],
                   class_correct[i], class_total[i]))
        else:
            print('Test accuracy of %5s: N/A (no training examples)' % str(i))

    print('\nTest Accuracy (Overall): %2.2f%% (%2d/%2d)' %
          (100 * sum(class_correct) / sum(class_total),
           sum(class_correct), sum(class_total)))

if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()

    # define custom flags
    # ('--name', 'default value', 'dtype', 'description comment of flag')
    parser.add_argument('--batch_size', default='64', type=int, help='batch_size for training')
    parser.add_argument('--num_epochs', default='40', type=int, help='number of training epochs')
    parser.add_argument('--scheduler', default='stepLR_plateau', type=str,
                        help='learning rate scheduler, pick between stepLR and exponential')
    parser.add_argument('--steps', default='1', type=int, help='how many steps to take - only for stepLR')
    parser.add_argument('--initial_lr', default='0.001', type=float, help='initial learning rate')
    parser.add_argument('--final_lr', default='5e-7', type=float, help='final learning rate')
    parser.add_argument('--optimizer', choices=['SGD', 'Adam'], default='SGD', type=str,
                        help='one of adam or sgd optimizers')
    parser.add_argument('--weight_decay', default='0.0', type=float, help='weight decay for optimizer')
    parser.add_argument('--momentum', default='0.9', type=float, help='momentum for optimizer - SGD only')
    parser.add_argument('--eval_every_n_epochs', default=1, type=int, help='how often to run validation')
    parser.add_argument('--print_freq', default=50, type=int, help='how often to print training log (steps)')
    parser.add_argument('--num_workers', default=4, type=int, help='how many threads to use on CPU')

    parser.add_argument('--classes', nargs='+', default=list(range(10)), type=int, help='list of used classes')

    # # Perturbation arguments
    parser.add_argument('--k', default=30.0, type=float, help='frequency of the perturbed signal')
    parser.add_argument('--epsilon', default=0.5, type=float, help='amplitude of the perturbed signal')
    parser.add_argument('--padding', default=0.0, type=float, help='signal padding')
    parser.add_argument('--vec_ind', nargs='+', default=[0], type=int, help='PCA vector index (0-9 for now)')
    parser.add_argument('--shape', default='cosine', type=str,
                        help='shape of the perturbed signal (cosine, sawtooth)')
    parser.add_argument('--model', default='resnet18', type=str,
                        help='Student model (resnet18)')
    parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset (cifar10 or fmnist)')
    parser.add_argument('--filename', default='student_model', type=str,
                        help='string used as a file name for the student model')
    parser.add_argument('--teacher_filename', nargs='+', default=['wm_teacher'], type=str,
                        help='string used as a file name for the teacher model')
    parser.add_argument('--train_part', default=1, type=int,
                        help='what partition of the training set to use (0, 1, or both)')
    parser.add_argument('--method', nargs='+', default='ours', type=str,
                        help='what method to use (ours, fingerprint, dawn, or ewe)')
    parser.add_argument('--dawn_prob', default=0.005, type=float, help='Probability of DAWN watermark')
    parser.add_argument('--root', default='models/', type=str, help='root of the experiments')
    parser.add_argument('--path', default='', type=str, help='path of the saved model, from 10class directory')
    parser.add_argument('--teacher_path', default='', type=str,
                        help='path of the teacher model, from 10class directory')
    parser.add_argument('--root_data', default='datasets/', type=str, help='root of the datasets')

    flags, unparsed = parser.parse_known_args()

    batch_size = flags.batch_size
    num_epochs = flags.num_epochs
    scheduler = flags.scheduler
    steps = flags.steps
    initial_lr = flags.initial_lr
    final_lr = flags.final_lr
    optimizer = flags.optimizer
    weight_decay = flags.weight_decay
    momentum = flags.momentum
    eval_every_n_epochs = flags.eval_every_n_epochs
    print_freq = flags.print_freq
    num_workers = flags.num_workers
    classes = flags.classes

    # Perturbation parameters
    k = flags.k
    epsilon = flags.epsilon
    padding = flags.padding
    vec_ind = flags.vec_ind
    shape = flags.shape
    model = flags.model
    dataset = flags.dataset
    filename = flags.filename
    teacher_filename = flags.teacher_filename
    train_part = flags.train_part
    method = flags.method
    root = flags.root
    path = flags.path
    teacher_path = flags.teacher_path
    root_data = flags.root_data

    train(None, None,
          batch_size, num_epochs, scheduler, steps, initial_lr, final_lr, optimizer, weight_decay,
          momentum, eval_every_n_epochs, print_freq, num_workers, classes,
          k, epsilon, padding, vec_ind, shape, model, dataset, filename, teacher_filename, train_part, method,
          root, path, teacher_path, root_data)
    print('total time: ', round(time.time() - total_start, 1), ' seconds')
