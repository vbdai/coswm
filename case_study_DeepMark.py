import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from utils import load_model, softmax_signal, build_periodogram
import argparse


def infer_train_data(epsilon=0.05, k=30.0, N=10, num_points_test=1000, target_class=0, train_part=0, num_workers=4,
                     vec_ind=0, plot_ind=0,
                     filename='student0', teacher_filename='source0',
                     root='models/', path='', teacher_path='', root_data='datasets/'):
    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set = torchvision.datasets.CIFAR10(root=root_data + 'cifar10/', download=True, transform=train_transform)

    # The train dataset is split in half
    idx = np.array(train_set.targets) == target_class
    if train_part == 0:
        idx = idx & (np.cumsum(idx) <= np.sum(idx) // 2)
    elif train_part == 1:
        idx = idx & (np.cumsum(idx) > np.sum(idx) // 2)

    train_set.targets = np.array(train_set.targets)[idx]
    train_set.data = np.array(train_set.data)[idx]

    idx = np.random.choice(train_set.targets.shape[0], num_points_test, replace=False)
    train_set.targets = np.array(train_set.targets)[idx]
    train_set.data = np.array(train_set.data)[idx]

    train_loader = DataLoader(
        train_set,
        batch_size=num_points_test,
        num_workers=num_workers,
    )

    lin_map = torch.tensor(np.loadtxt('rand_map_cifar10.csv'))
    lin_map0 = lin_map[:, vec_ind].view(lin_map.shape[0], 1).float()
    lin_map1 = lin_map[:, plot_ind].view(lin_map.shape[0], 1).float()

    teacher = load_model(root + path, teacher_filename, num_classes=N, offset=13)
    model = load_model(root + teacher_path, filename, num_classes=N, offset=0)
    teacher.eval()
    model.eval()

    for images, labels in train_loader:
        with torch.no_grad():
            output = F.softmax(model(images), dim=1)
            teacher_output = softmax_signal(teacher(images), images, k=k, epsilon=epsilon, num_classes=N, linear=True,
                                            lin_map=lin_map0)
            images = images.view([images.shape[0], images.shape[1] * images.shape[2] * images.shape[3]])

            idx_high_score = torch.topk(output[:, target_class], int(num_points_test*0.75), dim=0).indices.sort().values

            output = output[idx_high_score]
            teacher_output = teacher_output[idx_high_score]
            images = images[idx_high_score]

            xy_tensor = torch.cat([torch.matmul(images, lin_map1),
                                   output[:, 0].reshape([images.shape[0], 1]),
                                   teacher_output[:, 0].reshape([images.shape[0], 1])],
                                  1)

        xy_array = np.asarray(xy_tensor)

        freqs_array, _ = build_periodogram(xy_array, k=k)

    return xy_array, freqs_array


def generate_plots(xy, freqs, k=30.0):
    from matplotlib import pyplot

    x_train_norm = xy[:, 0]
    y_predict_0 = xy[:, 1]
    y_teacher_0 = xy[:, 2]

    fig2, ax = pyplot.subplots(ncols=2, nrows=1)
    fig2.set_size_inches(6.0, 3.0)
    fig2.subplots_adjust(wspace=2.0)
    ax[0].scatter(x_train_norm, y_teacher_0, marker='o', s=1.5, label='teacher')
    ax[0].scatter(x_train_norm, y_predict_0, marker='o', s=1.5, label='student')
    ax[0].set_xlabel(r'$p$', fontsize=18)
    ax[0].set_ylabel(r'$q_{i^*}$', fontsize=18)
    ax[0].set_ylim([0.6, 1])
    ax[0].legend(markerscale=4.0)

    y = freqs[:, 1]

    ax[1].scatter(freqs[:, 0] * 2 * np.pi, y, marker='o', s=1.5, label='student', color="orange")
    ax[1].axvline(x=k, ymin=0, ymax=1, linestyle='dotted', color='black', linewidth=2)
    ax[1].set_ylabel(r'$P(f)$', fontsize=18)
    ax[1].set_xlabel(r'$f$', fontsize=18)
    ax[1].set_xscale('log')
    ax[1].legend(markerscale=4.0, loc='lower left')

    fig2.canvas.draw()
    pyplot.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # define custom flags
    # ('--name', 'default value', 'dtype', 'description comment of flag')
    parser.add_argument('--classes', nargs='+', default=list(range(10)), type=int, help='list of used classes')

    # Perturbation arguments
    parser.add_argument('--k', default=30.0, type=float, help='frequency of the perturbed signal')
    parser.add_argument('--epsilon', default=0.05, type=float, help='amplitude of the perturbed signal')
    parser.add_argument('--vec_ind', default=0, type=int, help='Projection vector index used for the watermark (0-9)')
    parser.add_argument('--plot_ind', default=1, type=int,
                        help='Projection vector index used for plots (non-matching projection) (0-9)')
    parser.add_argument('--num_points', default=1000, type=int, help='Number of sampled points on the plots')
    parser.add_argument('--train_part', default=1, type=int, help='Using teacher (0) or student (1) training set')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of computing processes')
    parser.add_argument('--filename', default='wm_student', type=str,
                        help='string used as a file name for the student model')
    parser.add_argument('--teacher_filename', default='wm_teacher', type=str,
                        help='string used as a file name for the teacher model')
    parser.add_argument('--root', default='models/', type=str, help='root of the experiments')
    parser.add_argument('--path', default='', type=str, help='path of the saved model, from 10class directory')
    parser.add_argument('--teacher_path', default='', type=str,
                        help='path of the teacher model, from 10class directory')
    parser.add_argument('--root_data', default='datasets/', type=str, help='root of the datasets')
    flags, unparsed = parser.parse_known_args()

    k = flags.k  # Frequency of the signal
    epsilon = flags.epsilon  # Amplitude of the signal
    vec_ind = flags.vec_ind
    plot_ind = flags.plot_ind
    num_points = flags.num_points
    num_classes = 10  # Number of classes
    train_part = flags.train_part
    num_workers=flags.num_workers
    root = flags.root
    path = flags.path
    teacher_path = flags.teacher_path
    root_data = flags.root_data

    filename = flags.filename + '.pth'
    teacher_filename = flags.teacher_filename + '.pth'

    xy_match, freqs_match = \
        infer_train_data(epsilon=epsilon, k=k, N=num_classes, num_points_test=num_points, train_part=train_part,
                         vec_ind=vec_ind, plot_ind=vec_ind, num_workers=num_workers,
                         filename=filename, teacher_filename=teacher_filename,
                         root=root, path=path, teacher_path=teacher_path, root_data=root_data)

    generate_plots(xy_match, freqs_match, k)

    xy_non_match, freqs_non_match = \
        infer_train_data(epsilon=epsilon, k=k, N=num_classes, num_points_test=num_points, train_part=train_part,
                         vec_ind=vec_ind, plot_ind=plot_ind, num_workers=num_workers,
                         filename=filename, teacher_filename=teacher_filename,
                         root=root, path=path, teacher_path=teacher_path, root_data=root_data)

    generate_plots(xy_non_match, freqs_non_match, k)
