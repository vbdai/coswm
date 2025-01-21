# DeepMark: Robust Model Watermarking Against Ensemble Distillation Attacks
## Accompanying code exerpts

We include some code files for the purpose of result reproductibility, which we describe here.

## Requirements
Here are the package versions we used
* pytorch: 1.3.0
* torchvision: 0.4.1
* numpy: 1.19.2
* pandas: 1.0.3
* astropy: 3.2.3
* matplotlib: 3.2.1

## Script Descriptions
* `case_study_DeepMark.py`: Reproduces Figure 2 (b) and (c) results,
* `case_study_no_wm.py`: Reproduces Figure 2 (a) results, requires training or downloading unwatermarked models, see below for instructions,
* `train_teacher.py`: Trains a watermarked model,
* `train_student.py`: Trains a student of another model through distillation,
* `wm_extract.py`: Evaluates a student model and returns its accuracy, its teacher accuracy, and its extracted watermark strength,
* `utils.py`: Contains many utility functions used in the above scripts,
* `models.py`: Contains the ResNet models we use in the scripts.

## Instructions

### Case Study Results
`case_study_DeepMark.py` can be run directly with the default arguments and provided models.

`case_study_no_wm.py` needs unwatermarked models, which can be obtained via two methods
1. From the online link [https://drive.google.com/drive/folders/1PqKAEEs8Tx-Xk23PmwIsqcqIWukII7hB]. Model files should go in the `models/` folder
2. By running the provided scripts to train the models as follows
```
python train_teacher.py --epsilon 0.0 --filename no_wm_teacher
python train_student.py --epsilon 0.0 --teacher_filename no_wm_teacher --filename no_wm_student
```
then running the `case_study_no_wm.py` script with default values.

For the two scripts `train_teacher.py` and `train_student.py` we have many customization options available, the main ones are listed here
* `k`: sets the angular frequency of the signal function ($`f`$ in the paper),
* `epsilon`: sets the amplitude parameter of the signal function,
* `vec_ind`: sets one of the precomputed projection vectors from `rand_map_cifar10.csv` provided,
* `filename`: sets the output model filename (do not include `.pth`),
* `teacher_filename`: Sets all teacher model filenames (do not include `.pth`, list filenames separated by a space for multiple models), watermarked teachers must be listed before unwatermarked teachers,
* `method`: Should be a list of the same length as `teacher_filename`, indicates which models were watermarked by DeepMark by inputting `'ours'` at its position in the list, otherwise inputting `'indep'` for an unwatermarked teacher model.

The script `wm_extract.py` should match the parameters listed above for the training scripts to obtained the desired results.
Additional options:
* `proj_vec`: list of all projection vector indices to evaluate,
* `vec_eval`: list of all projection vectors associated to each watermarked teacher model.
Watermark strengths are returned in the format of a lost for all projections listed in `proj_vec`.
