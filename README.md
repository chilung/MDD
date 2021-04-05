# Margin Disparity Discrepancy

## Prerequisites:

* Python3
* PyTorch ==0.3.1 (with suitable CUDA and CuDNN version)
* torchvision == 0.2.0
* Numpy
* argparse
* PIL
* tqdm
* pyyaml
* easydict
* sklearn

## Dataset:

You need to modify the path of the image in every ".txt" in "./data".

## Training:

You can run "./scripts/train.sh" to train and evaluate on the task. Before that, you need to change the project root, dataset (Office-Home or Office-31), data address and CUDA_VISIBLE_DEVICES in the script.

## Citation:

If you use this code for your research, please consider citing:

```
@inproceedings{MDD_ICML_19,
  title={Bridging Theory and Algorithm for Domain Adaptation},
  author={Zhang, Yuchen and Liu, Tianle and Long, Mingsheng and Jordan, Michael},
  booktitle={International Conference on Machine Learning},
  pages={7404--7413},
  year={2019}
}
```
## Contact
If you have any problem about our code, feel free to contact zhangyuc17@mails.tsinghua.edu.cn.

## Study Progress
```
train.py: 105 def train_batch: 107 model_instance.get_loss
MDD.py: class MDD: get_loss
labels_source:32, input:64, outputs:64, outputs_adv:64

outputs.src -------> classifier_loss <------- labels_source
outputs_adv.src -------> classifier_loss_adv_src <------- outputs->target_adv.src
outputs_adv.tgt -------> classifier_loss_adv_tgt <------- outputs->target_adv.tgt
transfer_loss = weight*classifier_loss_adv_src + classifier_loss_adv_tgt
total_loss = classifier_loss + transfer_loss

forward:
fearutes --> (classifier_layer) --> outputs --> (softmax) --> softmax_outputs
         \-> (grl_layer) --> features_adv --> (classifier_layer_2) --> outputs_adv
```
