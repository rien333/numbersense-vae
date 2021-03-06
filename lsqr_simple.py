from lmfit import minimize, Parameters, report_fit
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
import random

DATA_W = 161
DATA_H = 161
C = 5 # amount of class
nan_eps = 1e-6 # log(0) does not exist

# idk what to do with epsilon but maybe just add a small constant to the result
def residual(params, cum_area, N, activation, eps):
    bias = params['B0'].value # number of objects coefficient
    b1 = params['B1'].value # number of objects coefficient
    b2 = params['B2'].value # cum. area size coefficient
    
    # Normalize cum_area and N
    # N_norm = N / params['B1'].max
    # cum_area_norm = cum_area / params['B2'].max
    
    # atm, this data is already normalized somehwere 
    N_norm = (N + nan_eps) / (C+nan_eps)
    cum_area_norm = (cum_area+nan_eps) / ((DATA_H*DATA_W) + nan_eps)

    # print(N)
    # print(cum_area_norm)
    # residual between model fit and ground truth neuron activations
    model = b1 * np.log(N_norm) + b2 * np.log(cum_area_norm) + bias
    # return (activation - model) / eps
    return activation - model

params = Parameters()
# You could give some of the values @stoianov2012 report as initial values
# idk if these min and max apply to the coefficients but maybe?

# params.add('B1', value=0.7, min=0.0, max=(4.0+N_eps)) # Number of objects
# params.add('B2', value=0.7, min=0.0, max=DATA_H*DATA_W) # Area
params.add('B0', value=0.01, min=-2.0, max=2.0) # bias
params.add('B1', value=0.03, min=-2.0, max=2.0) # Number of objects
params.add('B2', value=0.03, min=-2.0, max=2.0) # Area
eps = 5e-5

# High neurons should conversely collerate with *either* cum_a or *N*, but (almost?) never with both
samples = 992500
neurons = 8
# It's actually somewhat less probable that so many neurons are redundant
noisy_neurons = 0.2
obj_s = (DATA_W*DATA_H) * 0.15
size_std = 0.35 # was 0.35
# act_std = 0.08
act_std = 0.12 # was 12

N = np.random.randint(low=0, high=5, size=samples)
N_norm = N / C
# It's probably a good idea to add at least a little to this as some neurons that code for size
# are often zero, which is never true in the VAEs case
# maybe reflect this more in the activation tho
cum_area = N * obj_s * np.random.normal(1.0, size_std, size=samples)
cum_area = np.clip(cum_area, 0, a_max=None)
cum_area_norm = cum_area / (DATA_W*DATA_H)

# neurons of interest
noi = np.array(random.sample(range(neurons), round(neurons*(1-noisy_neurons))))
np.random.shuffle(noi)
# area and number neurons
n_a, n_n = np.split(noi, 2)

np.set_printoptions(precision=3, linewidth=120, suppress=True, edgeitems=90)
activations = np.random.normal(0.5, act_std, size=(samples, neurons))
# print(activations)
# neurons do some other stuff as well tho so it makes sense that they sometimes are non-zero
# (they can be zero for N=0), so add some stuff
add_std = 0.00
enc_std = 0.011
activations[:, n_n] += (N_norm * np.random.normal(1.32, enc_std, size=samples)).reshape(samples, 1)
activations[:, n_a] += (cum_area_norm * np.random.normal(1.32, enc_std, size=samples)).reshape(samples, 1)
# activations[:, n_a] += np.random.normal(0, 0.07)
# activations[:, n_a] *= cum_area 
# for s in zip(activations, cum_area, N):
#     s[n_a] *=  
#     s[n_n] *= 

#     s[n_a] += np.random.normal(0, 0.07)
#     s[n_n] += np.random.normal(0, 0.07)

# print("\n",activations)
# for v in (N, cum_area):
#     print('[%s]' % (' '.join('%5.f' % i for i in v)))

# fig = plt.figure(figsize=(20, 16))
# plt.scatter(N, cum_area)
# plt.xlabel('Number of objecs')
# plt.ylabel('Cumaltative area')
# plt.show()


# fig = plt.figure()
# for i in range(neurons):
#     ax = fig.add_subplot(2, int(neurons/2), i+1, projection='3d')
#     ax.scatter(N, cum_area, activations[:, i], c=np.random.random(size=(1,3)))

#     if i in n_a:
#         n_type = "Area"
#     elif i in n_n:
#         n_type = "Numerosity"
#     else:
#         n_type = "Other"

#     ax.set_title('N%s - %s' % (i+1, n_type))
#     ax.set_xlabel('N')
#     ax.set_ylabel('A')
#     ax.set_zlabel('R')    
# plt.show()

# convert to label ish values
# N *= (C+1)
# N = N.astype(np.uint8).astype(np.float)
# cum_area *= (DATA_W*DATA_H)
# cum_area = np.round(cum_area)
# print("Activation data:\n", activation)
# print("N:\n", N)
# print("Cumelative area:\n", cum_area)
# exit(0)

nia_idx = random.randint(0, len(n_a)-1)
nia = n_a[nia_idx]
print("Minimizing area neuron", nia)
nin_idx = random.randint(0, len(n_n)-1)
nin = n_n[nin_idx]
print("Minimizing numerosity neuron", nin)

n_o = np.setdiff1d(np.arange(neurons), noi)
nio_idx = random.randint(0, len(n_o)-1)
nio = n_o[nio_idx]
print("Minimizing other neuron", nio)


# print(activations)
# print(activations[:, nin])
# exit(0)
# Least squares optimization
a = activations[:, nin]
out = minimize(residual, params,  args=(cum_area, N, a, eps),)
report_fit(out)

print("R^2:",  1 - (np.sum((out.residual)**2) / np.sum((a - np.mean(a))**2)))
print("R^2:",  1 - out.residual.var() / np.var(a))
