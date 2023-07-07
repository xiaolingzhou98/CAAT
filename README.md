# CAAT
 
This is the code for the paper: Combining Adversaries with Anti-adversaries in Training<br>

Setups
-------  
The requiring environment is as bellow:<br>
* Linux<br>
* python 3.8<br>
* pytorch 1.9.0<br>
* torchvision 0.10.0<br>

Running CAAT on benchmark datasets (CIFAR-10).
-------  
Here are two examples for training imbalanced and noisy data:<br>
ResNet32 on CIFAR10-LT with imbalanced factor of 10:<br>

`python CAAT.py --dataset cifar10 --imbalanced_factor 10`

ResNet32 on noisy CIFAR10 with 20\% pair-flip noise:<br>
`python CAAT.py --dataset cifar10 --corruption_type flip2 --corruption_ratio 0.2`
