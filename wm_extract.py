import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.nn import MSELoss
import numpy as np
import pandas as pd
import argparse
from utils import softmax_signal, build_periodogram, CEProbs, load_model, \
                  get_spectrum_window, select_split_classes


def eval_ours(train_set=None, test_set=None,
              num_workers=4, epsilon=0.05, k=0.3, target_class=0, N=10, num_points_test=1000,
              model_arc='resnet18', dataset='cifar10', train_part=1, proj_vec=[0], vec_eval=[0],
              columns=['model_name'], threshold=8.0, method='[ours]', filename='', teacher_filename=[''],
              root='models/', path='', teacher_path='', root_data='datasets/'):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if dataset == 'fmnist':
        # Loading dataset
        if train_set is None:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            train_set = torchvision.datasets.FashionMNIST(root=root_data + 'fmnist/', download=False,
                                                          transform=train_transform)
            test_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            test_set = torchvision.datasets.FashionMNIST(root=root_data + 'fmnist/', download=False,
                                                         train=False, transform=test_transform)
            # See teacher script for explanation of the dataset split
            # (inverted inequalities to get other half of the split)
            select_split_classes(train_set, test_set, [target_class], train_part, dataset=dataset, full_test=True)
    else:
        # cifar10 by default
        # Loading dataset if none provided
        if train_set is None:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            train_set = torchvision.datasets.CIFAR10(root=root_data + 'cifar10/', download=False,
                                                     transform=train_transform)
            test_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            test_set = torchvision.datasets.CIFAR10(root=root_data + 'cifar10/', download=False,
                                                    train=False, transform=test_transform)
            select_split_classes(train_set, test_set, [target_class], train_part, dataset=dataset, full_test=True)


    train_loader = DataLoader(
        train_set,
        batch_size=num_points_test,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set,
        batch_size=128,
        num_workers=num_workers
    )

    df_output = pd.DataFrame(columns=columns)

    # Path and filename strings
    teacher_path = root + teacher_path
    model_path = root + path

    lin_map = []
    for vec in proj_vec:
        map_temp = torch.tensor(np.loadtxt('rand_map_%s.csv' % dataset)).to(device)
        lin_map.append(map_temp[:, vec].view(map_temp.shape[0], 1).float())

    teachers = []
    for fname in teacher_filename:
        teacher = load_model(teacher_path, fname + '.pth', num_classes=N, dataset=dataset, model_arc=model_arc,
                             old=0, device=device, offset=0)
        teacher.eval()
        teachers.append(teacher)

    model = load_model(model_path, filename + '.pth', num_classes=N, old=0, dataset=dataset, model_arc=model_arc,
                       device=device, offset=0)
    model.eval()

    teacher_acc = []
    student_acc_test = []

    # For Testing accuracy
    ce_loss_total = 0
    mse_loss_total = 0
    class_correct_student = list(0 for i in range(N))
    class_total_student = list(0 for i in range(N))
    class_correct_teacher = list(0 for i in range(N))
    class_total_teacher = list(0 for i in range(N))

    for images, labels in test_loader:
        with torch.no_grad():
            images = images.to(device)
            labels = labels.to(device)

            output = F.softmax(model(images), dim=1)
            teacher_output = torch.zeros([images.shape[0], N]).to(device)
            for i in range(len(method)):
                if method[i] == 'ours':
                    teacher_output += softmax_signal(teachers[i](images), images, k=k, epsilon=epsilon, num_classes=N,
                                                     linear=True, lin_map=lin_map[vec_eval[i]]) / len(method)
                else:
                    teacher_output += F.softmax(teachers[i](images), dim=1) / len(method)

            mse = MSELoss()
            ce = CEProbs(num_classes=N)
            mse_loss = mse(F.softmax(output, dim=1), teacher_output).item()
            ce_loss = ce(output, teacher_output).item()
            mse_loss_total += mse_loss * images.size(0)
            ce_loss_total += ce_loss * images.size(0)

            _, predictions = torch.max(output, 1)
            # Compare with ground truth
            correct = (predictions == labels).squeeze()
            # Test accuracy computation
            for i in range(len(labels)):
                lab = labels[i]
                class_correct_student[lab] += correct[i].item()
                class_total_student[lab] += 1

            _, predictions = torch.max(teacher_output, 1)
            # Compare with ground truth
            correct = (predictions == labels).squeeze()
            # Test accuracy computation
            for i in range(len(labels)):
                lab = labels[i]
                class_correct_teacher[lab] += correct[i].item()
                class_total_teacher[lab] += 1

    print('-----Testing-----')
    mse_loss_total = mse_loss_total / len(test_set)
    ce_loss_total = ce_loss_total / len(test_set)
    print('MSE Loss: %2.4f' % mse_loss_total)
    print('CE Loss:  %2.4f' % ce_loss_total)
    print()

    print('Student Accuracy:')

    acc = 100 * sum(class_correct_student) / sum(class_total_student)
    print('Accuracy (Overall): %2.2f%% (%2d/%2d)' %
          (acc, sum(class_correct_student), sum(class_total_student)))
    student_acc_test.append(acc)

    print()

    print('Teacher Accuracy:')

    acc = 100 * sum(class_correct_teacher) / sum(class_total_teacher)
    print('Accuracy (Overall): %2.2f%% (%2d/%2d)' %
          (acc, sum(class_correct_teacher), sum(class_total_teacher)))
    teacher_acc.append(acc)

    images, labels = next(iter(train_loader))
    images = images.to(device)
    with torch.no_grad():
        output = F.softmax(model(images), dim=1)
        teacher_output = torch.zeros([images.shape[0], N]).to(device)
        for i in range(len(method)):
            if method[i] == 'ours':
                teacher_output += softmax_signal(teachers[i](images), images, k=k, epsilon=epsilon, num_classes=N,
                                                 linear=True, lin_map=lin_map[vec_eval[i]]) / len(method)
            else:
                teacher_output += F.softmax(teachers[i](images), dim=1) / len(method)

        images_flat = images.view([images.shape[0], images.shape[1] * images.shape[2] * images.shape[3]])

        idx_high_score = output[:, target_class] > 1.0 * torch.median(output[:, target_class])

        output = output[idx_high_score]
        teacher_output = teacher_output[idx_high_score]
        images_flat = images_flat[idx_high_score]

        new_psnr_list = []
        for i in proj_vec:
            xy_tensor = torch.cat([torch.matmul(images_flat, lin_map[i]),
                                   output[:, target_class].reshape([images_flat.shape[0], 1]),
                                   teacher_output[:, target_class].reshape([images_flat.shape[0], 1])],
                                  1)
            xy_array = np.asarray(xy_tensor.cpu())

            freqs_array, thetas = build_periodogram(xy_array, k=k)

            win005, new_psnr = get_spectrum_window(freqs_array[:, 0], freqs_array[:, 1], k, halfwidth=0.005)
            new_psnr_list.append(new_psnr)

    ground_truth = vec_eval

    pred_list = []
    for i in proj_vec:
        if new_psnr_list[i] > threshold:
            pred_list.append(i)

    filename_out = filename.split('.')[0]

    student_acc_array = np.array(student_acc_test)
    teacher_acc_array = np.array(teacher_acc)

    out_list = [filename_out, student_acc_array.mean(), teacher_acc_array.mean(),
                new_psnr_list, pred_list, ground_truth]

    return df_output.append(dict(zip(list(df_output), out_list)), ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # define custom flags
    # ('--name', 'default value', 'dtype', 'description comment of flag')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers')
    parser.add_argument('--epsilon', default=0.05, type=float, help='Signal amplitude coefficient')
    parser.add_argument('--k', default=30.0, type=float, help='Signal angular frequency')
    parser.add_argument('--num_classes', default=10, type=int, help='Number of classes')
    parser.add_argument('--num_points_test', default=200, type=int, help='Number test points')
    parser.add_argument('--model_arc', default='resnet18', type=str, help='Model architecture')
    parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset ("cifar10" or "fmnist")')
    parser.add_argument('--teacher_filename', nargs='+', default=['wm_teacher'], type=str,
                        help='Filenames of all teacher models')
    parser.add_argument('--filename', default='wm_student', type=str, help='Student model filename')
    parser.add_argument('--target_class', default=0, type=int, help='Class of generated samples')
    parser.add_argument('--train_part', default=1, type=int, help='Part of the training data used for training')
    parser.add_argument('--proj_vec', nargs='+', default=[0], type=int,
                        help='List of projection vector index used by the teacher models')
    parser.add_argument('--vec_eval', nargs='+', default=[0], type=int, help='Index of proj_vec used for evaluation')
    parser.add_argument('--method', nargs='+', default=['ours'], type=str, help='Methods used by the teacher models')
    parser.add_argument('--path', default='', type=str, help='Specific folder for student models')
    parser.add_argument('--teacher_path', default='', type=str, help='Specific folder for teacher models')
    parser.add_argument('--root', default='models/', type=str, help='root of the experiments')
    parser.add_argument('--root_data', default='datasets/', type=str, help='root of the datasets')

    parser.add_argument('--crit_metric', default='new_psnr', type=str, help='Metric used for identification')
    parser.add_argument('--threshold', default=8.0, type=float, help='Threshold value for identification')


    flags, unparsed = parser.parse_known_args()

    epsilon = flags.epsilon  # Amplitude of the signal
    k = flags.k  # Frequency of the signal
    N = flags.num_classes  # Number of classes
    num_points_test = flags.num_points_test  # Number of points
    model_arc = flags.model_arc
    dataset = flags.dataset
    num_workers = flags.num_workers
    target_class = flags.target_class
    train_part = flags.train_part
    proj_vec = flags.proj_vec
    vec_eval = flags.vec_eval
    method = flags.method
    path = flags.path
    teacher_path = flags.teacher_path
    filename = flags.filename
    teacher_filename = flags.teacher_filename
    root = flags.root
    root_data = flags.root_data

    crit_metric = flags.crit_metric
    threshold = flags.threshold

    cols = ['model_name', 'student_acc_test', 'teacher_acc_test', 'new_psnr', 'pred', 'ground_truth']

    df_final_student = pd.DataFrame(columns=cols)
    df_final_indep = pd.DataFrame(columns=cols)

    df_out = eval_ours(None, None,
                       num_workers=num_workers, epsilon=epsilon, k=k, target_class=target_class, N=N,
                       num_points_test=num_points_test, model_arc=model_arc, dataset=dataset, train_part=train_part,
                       proj_vec=proj_vec, vec_eval=vec_eval, columns=cols, threshold=threshold,
                       method=method, filename=filename, teacher_filename=teacher_filename,
                       root=root, path=path, teacher_path=teacher_path,
                       root_data=root_data)

    print()
    print()
    print('-----Summary Statistics-----')
    print()
    print('Student name: %s' % df_out.model_name[0])
    print('Student Accuracy: %f' % df_out.student_acc_test[0])
    print('Teacher Ensemble Accuracy: %f' % df_out.teacher_acc_test[0])
    print('Signal Strengths: %s' % df_out.new_psnr[0])
    print('Predicted Teachers: %s' % df_out.pred[0])
    print('True Teachers: %s' % df_out.ground_truth[0])

    print('done')
