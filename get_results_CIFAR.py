import tensorflow as tf
#import numpy as np

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

from SimAnCifar100 import Do_Simulation as Do_Simulationcifar100


EPOCHS = 12

T = [1 for k in range(EPOCHS)]
version_title = "CIFAR100 Benchmark, T=1, {}.txt."
Do_Simulationcifar100(EPOCHS,T,version_title)

T = [1/2**k for k in range(EPOCHS)]
version_title = "cifar100 T=half, {}.txt"
Do_Simulationcifar100(EPOCHS,T,version_title)

T = [1/3**k for k in range(EPOCHS)]
version_title = "cifar100 T=third, {}.txt"
Do_Simulationcifar100(EPOCHS,T,version_title)

T = [1/4**k for k in range(EPOCHS)]
version_title = "cifar100 T=quarter, {}.txt"
Do_Simulationcifar100(EPOCHS,T,version_title)

T = [1/5**k for k in range(EPOCHS)]
version_title = "cifar100 T=fifth, {}.txt"
Do_Simulationcifar100(EPOCHS,T,version_title)

T = [1/5**k for k in range(EPOCHS)]
version_title = "cifar100 T=sixth, {}.txt"
Do_Simulationcifar100(EPOCHS,T,version_title)
