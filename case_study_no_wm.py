import argparse

from case_study_DeepMark import infer_train_data, generate_plots


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # define custom flags
    # ('--name', 'default value', 'dtype', 'description comment of flag')
    parser.add_argument('--classes', nargs='+', default=list(range(10)), type=int, help='list of used classes')

    # Perturbation arguments
    parser.add_argument('--k', default=30.0, type=float, help='frequency of the perturbed signal')
    parser.add_argument('--epsilon', default=0.0, type=float, help='amplitude of the perturbed signal')
    parser.add_argument('--vec_ind', default=0, type=int, help='Projection vector index used for the watermark (0-9)')
    parser.add_argument('--plot_ind', default=1, type=int,
                        help='Projection vector index used for plots (non-matching projection) (0-9)')
    parser.add_argument('--num_points', default=1000, type=int, help='Number of sampled points on the plots')
    parser.add_argument('--train_part', default=1, type=int, help='Using teacher (0) or student (1) training set')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of computing processes')
    parser.add_argument('--filename', default='no_wm_student', type=str,
                        help='string used as a file name for the student model')
    parser.add_argument('--teacher_filename', default='no_wm_teacher', type=str,
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
