# Towards Adversarial Robustness of Bayesian Neural Network through Hierarchical Variational Inference [[paper]](https://openreview.net/forum?id=Cue2ZEBf12)

Baseline of this code is [the official repository](https://github.com/xuanqing94/BayesianDefense) for this [paper](https://openreview.net/pdf?id=rk4Qso0cKm). We just replace the BNN regularizer from ELBO with enhanced Bayesian regularizer based on hierarchical-ELBO.

---

# Hierarchical-Bayeisan-Defense

## Dataset
+ CIFAR10
+ STL10
+ CIFAR100
+ Tiny-ImageNet

## Network
+ VGG16 (for CIFAR-10/CIFAR-100/Tiny-ImageNet)
+ Aaron (for STL10)
+ WideResNet (for CIFAR-10/100)

## Attack (by [torchattack](https://github.com/Harry24k/adversarial-attacks-pytorch))
+ PGD attack
+ EOT-PGD attack

## Defense methods
+ `adv`: Adversarial training
+ `adv_vi`: Adversarial training with Bayesian neural network
+ `adv_hvi`: Adversarial training with Enhanced Bayesian neural network based on hierarchical-ELBO


## How to Train

### 1. Adversarial training

Run `train_adv.sh`

```bash
lr=0.01
steps=10
max_norm=0.03
data=tiny # or `cifar10`, `stl10`, `cifar100`
root=./datasets
model=vgg # vgg for `cifar10` `stl10` `cifar100`, aaron for `stl10`, wide for `cifar10` or `cifar100`
model_out=./checkpoint/${data}_${model}_${max_norm}_adv
echo "Loading: " ${model_out}
CUDA_VISIBLE_DEVICES=0 python ./main_adv.py \
                        --lr ${lr} \
                        --step ${steps} \
                        --max_norm ${max_norm} \
                        --data ${data} \
                        --model ${model} \
                        --root ${root} \
                        --model_out ${model_out}.pth \
```

### 2. Adversarial training with BNN

Run `train_adv_vi.sh`

```bash
lr=0.01
steps=10
max_norm=0.03
sigma_0=0.1
init_s=0.1
data=tiny # or `cifar10`, `stl10`, `cifar100`
root=./datasets
model=vgg # vgg for `cifar10` `stl10` `cifar100`, aaron for `stl10`, wide for `cifar10` or `cifar100`
model_out=./checkpoint/${data}_${model}_${max_norm}_adv_vi
echo "Loading: " ${model_out}
CUDA_VISIBLE_DEVICES=0 python3 ./main_adv_vi.py \
                        --lr ${lr} \
                        --step ${steps} \
                        --max_norm ${max_norm} \
                        --sigma_0 ${sigma_0} \
                        --init_s ${init_s} \
                        --data ${data} \
                        --model ${model} \
                        --root ${root} \
                        --model_out ${model_out}.pth \
```

### 3. Adversarial training with enhanced Bayesian regularizer based on hierarchical-ELBO

Run `train_adv_hvi.sh`

```bash
lr=0.01
steps=10
max_norm=0.03
sigma_0=0.1
init_s=0.1
data=tiny # or `cifar10`, `stl10`, `cifar100`
root=./datasets
model=vgg # vgg for `cifar10` `stl10` `cifar100`, aaron for `stl10`, wide for `cifar10` or `cifar100`
model_out=./checkpoint/${data}_${model}_${max_norm}_adv_hvi
echo "Loading: " ${model_out}
CUDA_VISIBLE_DEVICES=0 python3 ./main_adv_hvi.py \
                        --lr ${lr} \
                        --step ${steps} \
                        --max_norm ${max_norm} \
                        --sigma_0 ${sigma_0} \
                        --init_s ${init_s} \
                        --data ${data} \
                        --model ${model} \
                        --root ${root} \
                        --model_out ${model_out}.pth \
```

## How to Test

### Testing adversarial robustness

Run `acc_under_attack.sh`

```bash
model=vgg # vgg for `cifar10` `stl10` `cifar100`, aaron for `stl10`, wide for `cifar10` or `cifar100`
defense=adv_hvi # or `adv_vi`, `adv`
data=tiny-imagenet # or `cifar10`, `stl10`, `cifar100`
root=./datasets
n_ensemble=50
step=10
max_norm=0.03
echo "Loading" ./checkpoint/${data}_${model}_${max_norm}_${defense}.pth

CUDA_VISIBLE_DEVICES=0 python3 acc_under_attack.py \
    --model $model \
    --defense $defense \
    --data $data \
    --root $root \
    --n_ensemble $n_ensemble \
    --step $step \
    --max_norm $max_norm

```

## How to check the learning parameters and KL divergence

Run `check_parameters.sh`

```bash
model=vgg # vgg for `cifar10` `stl10` `cifar100`, aaron for `stl10`, wide for `cifar10` or `cifar100`
defense=adv_hvi # or `adv_vi`
data=tiny-imagenet # or `cifar10`, `stl10`, `cifar100`
max_norm=0.03
echo "Loading" ./checkpoint/${data}_${model}_${max_norm}_${defense}.pth

CUDA_VISIBLE_DEVICES=0 python3 check_parameters.py \
    --model $model \
    --defense $defense \
    --data $data \
    --max_norm $max_norm \
```

## How to check uncertainty by predictive entropy

Run `uncertainty.sh`

```bash
model=vgg # vgg for `cifar10` `stl10` `cifar100`, aaron for `stl10`, wide for `cifar10` or `cifar100`
defense=adv_hvi # or `adv_vi`
data=tiny-imagenet # or `cifar10`, `stl10`, `cifar100`
root=./datasets
n_ensemble=50
step=10
max_norm=0.03
echo "Loading" ./checkpoint/${data}_${model}_${max_norm}_${defense}.pth

CUDA_VISIBLE_DEVICES=0 python3 uncertainty.py \
    --model $model \
    --defense $defense \
    --data $data \
    --root $root \
    --n_ensemble $n_ensemble \
    --step $step \
    --max_norm $max_norm
```