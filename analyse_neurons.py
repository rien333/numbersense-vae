from lmfit import minimize, Parameters, report_fit
import SOSDataset
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib

DATA_H = SOSDataset.DATA_H
DATA_W = SOSDataset.DATA_W
C = 4

N = np.load("misc/N.npy")
cum_area = np.load("misc/cum_area.npy")
activations = np.load("misc/activations.npy")
noi = np.load("misc/noi.npy")
noi = list(noi)
noi += [noi[-1]]
noi = [82, 2, 180, 77, 137, 136, 115, 91, 88, 18, 35]
# noi = [2]
print(N.shape)
print(cum_area.shape)
print(activations.shape)
# fig = plt.figure()

# for i, n in enumerate(noi):
#     # ax = fig.add_subplot(2, int((len(noi))/2), i+1, projection='3d')
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1, projection='3d')
#     ax.scatter(N, cum_area, activations[:, n], c=np.random.random(size=(1,3)))
#     ax.set_title('Neuron %s' % (n))
#     ax.set_xlabel('N')
#     ax.set_ylabel('A')
#     ax.set_zlabel('R')
#     plt.show()

nan_eps = 1e-6 # log(0) does not exist

N_norm = (N + nan_eps) / (C+nan_eps)
cum_area_norm = (cum_area+nan_eps) / ((DATA_H*DATA_W) + nan_eps)
bias = -2.00932487
b1 = 0.39342263
b2 = -0.09529530

model = b1 * np.log(N_norm) + b2 * np.log(cum_area_norm) + bias
noi = 77

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter(N_norm, cum_area_norm, activations[:, noi], c=np.random.random(size=(1,3)))
# ax.plot(N_norm, model)
ax.set_title('$z_{%s}$' % (noi))
ax.set_xlabel('N')
ax.set_ylabel('A')
ax.set_zlabel('R')
plt.show()
