import tensorflow as tf
#import numpy as np

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

from SimAnMNIST import Do_Simulation as Do_SimulationMNIST
from SimAnCifar100 import Do_Simulation as Do_Simulationcifar10


EPOCHS = 12


T = [1 for k in range(EPOCHS)]
version_title = "MNIST Benchmark, T=1, {}.txt"
Do_SimulationMNIST(EPOCHS,T,version_title)

T = [1/2**k for k in range(EPOCHS)]
version_title = "MNIST T=half, {}.txt"
Do_SimulationMNIST(EPOCHS,T,version_title)

T = [1/3**k for k in range(EPOCHS)]
version_title = "MNIST T=third, {}.txt"
Do_SimulationMNIST(EPOCHS,T,version_title)

T = [1/4**k for k in range(EPOCHS)]
version_title = "MNIST T=quarter, {}.txt"
Do_SimulationMNIST(EPOCHS,T,version_title)

T = [1/5**k for k in range(EPOCHS)]
version_title = "MNIST T=fifth, {}.txt"
Do_SimulationMNIST(EPOCHS,T,version_title)

T = [1/5**k for k in range(EPOCHS)]
version_title = "MNIST T=sixth, {}.txt"
Do_SimulationMNIST(EPOCHS,T,version_title)


###############################################################################


T = [1 for k in range(EPOCHS)]
version_title = "CIFAR10 Benchmark, T=1, {}.txt."
Do_Simulationcifar10(EPOCHS,T,version_title)

T = [1/2**k for k in range(EPOCHS)]
version_title = "cifar10 T=half, {}.txt"
Do_Simulationcifar10(EPOCHS,T,version_title)

T = [1/3**k for k in range(EPOCHS)]
version_title = "cifar10 T=third, {}.txt"
Do_Simulationcifar10(EPOCHS,T,version_title)

T = [1/4**k for k in range(EPOCHS)]
version_title = "cifar10 T=quarter, {}.txt"
Do_Simulationcifar10(EPOCHS,T,version_title)

T = [1/5**k for k in range(EPOCHS)]
version_title = "cifar10 T=fifth, {}.txt"
Do_Simulationcifar10(EPOCHS,T,version_title)

T = [1/5**k for k in range(EPOCHS)]
version_title = "cifar10 T=sixth, {}.txt"
Do_Simulationcifar10(EPOCHS,T,version_title)
