import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
import datetime

def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

def grad(x):
    l: int = len(x)
    s = np.arange(l)
    r = np.arange(l)
    for i in range(1, l+1):
        s[i-1] = ((y[i-1]**2)*math.exp(-x[i-1])/(beta**2)-1)*0.5
    r[0] = (x[0] - rho * x[1])/(sigma**2)
    r[l-1] = (x[l-1] - rho*x[l-2])/(sigma**2)
    for i in range(1, l-1):
        r[i] = (-rho*x[i+1]+(1+rho**2)*x[i]-rho*x[i-1])/(sigma**2)
    result = np.subtract(s, r)
    return result
####
def loglikeli(x):
    l = len(x)
    prod = norm.pdf(x[0], loc=0, scale=math.sqrt(initiVar))*norm.pdf(y[0], loc=0, scale=beta*math.exp(x[0]/2))
    for i in range(1, l):
        prod = prod*norm.pdf(x[i], loc=rho*x[i-1], scale=sigma)*norm.pdf(y[i], loc=0, scale=beta*math.exp(x[i]/2))
    result = math.log(prod)
    return result

###Sequential Importance Resampling
###initialization
N = 300  ###number of samples
T = 30  ###number of time steps
rho: float = 0.95###or 0.98
sigma = 1 ##or 0.15
beta: float = 0.5###or 0.65
initiVar = (sigma ** 2) / (1 - rho ** 2)  ##initial variance
true_x = np.load('true_x_layer1.npy')
y = np.load('y_layer1.npy')

x0 = np.random.normal(0, math.sqrt(initiVar), 1)
true_x = np.zeros([T, 1])  ###hidden states
y = np.zeros([T, 1])
true_x[0] = x0
y[0] = beta * math.exp(true_x[0] / 2) * np.random.normal(0, 1, 1)

###true tracjatory
for t in range(1, T):
    true_x[t] = rho * true_x[t - 1] + sigma * np.random.normal(0, 1, 1)
    y[t] = beta * math.exp(true_x[t - 1] / 2) * np.random.normal(0, 1, 1)

### adaptive resampling
x = np.random.rand(N, T)  ###sampling particles
xu = np.random.rand(N, T)  ####particles after resampling
q = np.random.rand(N, T)  ###normalised weights
qq = np.random.rand(N, T)  ###unnormalised weights
I = np.random.rand(N, T)  ###offsprings
R = np.random.rand(N, 1)  ###variance
x[:, 0] = np.random.normal(0, math.sqrt(initiVar), N)
for j in range(1, N + 1):
    R[j - 1] = (beta ** (2)) * math.exp(x[j - 1, 0])
for j in range(1, N + 1):
    qq[j - 1, 0] = math.exp(-0.5 * (R[j - 1] ** (-1)) * (y[0] ** 2)) / math.sqrt(2 * math.pi * R[j - 1])
q[:, 0] = qq[:, 0] / math.fsum(qq[:, 0])
# Re=rmultinom(1, N, prob = q[,1])
probs = q[:, 0].tolist()
ESS = (np.sum(q[:, 0] ** 2)) ** (-1)
thre = 0.5 * N
if ESS <= thre:
    for j in range(1, N + 1):
        A = np.random.multinomial(1, probs)
        A = A.tolist()
        A = A.index(1)
        xu[j - 1, 0] = x[A, 0]
    q[:, 0] = [1 / N] * N
else:
    xu[:, 0] = x[:, 0]
###update and prediction stages
for t in range(1, T):
    x[:, t] = rho * xu[:, t - 1] + sigma * np.random.normal(0, 1, N)
    for j in range(1, N + 1):
        R[j - 1] = (beta ** 2) * math.exp(x[j - 1, t])
    for j in range(1, N + 1):
        qq[j - 1, t] = q[j - 1, t - 1]*math.exp(-0.5 * (R[j - 1] ** (-1)) * (y[t] ** 2)) / math.sqrt(2 * math.pi * R[j - 1])
    q[:, t] = qq[:, t] / sum(qq[:, t])
    ###resampling step
    probs = q[:, t].tolist()
    ESS = (np.sum(q[:, t]**2))**(-1)
    thre = 0.5*N
    if ESS <= thre:
        for j in range(1, N + 1):
            A = np.random.multinomial(1, probs)
            A = A.tolist()
            A = A.index(1)
            xu[j - 1, t] = x[A, t]
            xu[j - 1, 0:t] = xu[A, 0:t]
        q[:, t] = [1 / N] * N
    else:
        xu[:, t] = x[:, t]

####dynamic resampling
x = np.random.rand(N, T)  ###sampling particles
xu = np.random.rand(N, T)  ####particles after resampling
q = np.random.rand(N, T)  ###normalised weights
qq = np.random.rand(N, T)  ###unnormalised weights
I = np.random.rand(N, T)  ###offsprings
R = np.random.rand(N, 1)  ###variance
x[:, 0] = np.random.normal(0, math.sqrt(initiVar), N)
for j in range(1, N + 1):
    R[j - 1] = (beta ** (2)) * math.exp(x[j - 1, 0])
for j in range(1, N + 1):
    qq[j - 1, 0] = math.exp(-0.5 * (R[j - 1] ** (-1)) * (y[0] ** 2)) / math.sqrt(2 * math.pi * R[j - 1])
q[:, 0] = qq[:, 0] / math.fsum(qq[:, 0])
sort = sorted(range(len(q[:, 0])), key=lambda e: q[e, 0])
probs = q[:, 0].tolist()
k = 0
ESS = (np.sum(q[:, 0] ** 2)) ** (-1)
thre = 5*N/10
while ESS <= thre:
    delta = q[sort[k], 0] - 1 / N
    q[sort[(k + 1):N], 0] = (1 + delta / sum(q[sort[(k + 1):N], 0])) * q[sort[(k + 1):N], 0]
    A = np.random.multinomial(1, probs)
    A = A.tolist()
    A = A.index(1)
    xu[sort[k], 0] = x[A, 0]
    q[sort[k], 0] = 1 / N
    ESS = (np.sum(q[:, 0] ** 2)) ** (-1)
    k = k + 1
xu[sort[k:N], 0] = x[sort[k:N], 0]
###update and prediction stages
for t in range(1, T):
    x[:, t] = rho * xu[:, t - 1] + sigma * np.random.normal(0, 1, N)
    for j in range(1, N + 1):
        R[j - 1] = (beta ** 2) * math.exp(x[j - 1, t])
    for j in range(1, N + 1):
        qq[j - 1, t] = q[j - 1, t - 1] * math.exp(-0.5 * (R[j - 1] ** (-1)) * (y[t] ** 2)) / math.sqrt(
            2 * math.pi * R[j - 1])
    q[:, t] = qq[:, t] / sum(qq[:, t])
    sort = sorted(range(len(q[:, t])), key=lambda e: q[e, t])
    probs = q[:, t].tolist()
    k = 0
    ESS = (np.sum(q[:, t] ** 2)) ** (-1)
    thre = 5*N/10
    while ESS <= thre:
        delta = q[sort[k], t] - 1 / N
        q[sort[(k + 1):N], t] = (1 + delta / sum(q[sort[(k + 1):N], t])) * q[sort[(k + 1):N], t]
        A = np.random.multinomial(1, probs)
        A = A.tolist()
        A = A.index(1)
        xu[sort[k], t] = x[A, t]
        xu[sort[k], 0:t] = xu[A, 0:t]
        q[sort[k], t] = 1 / N
        ESS = (np.sum(q[:, t] ** 2)) ** (-1)
        k = k + 1
    xu[sort[k:N], t] = x[sort[k:N], t]

####rejection resampling
x = np.random.rand(N, T)  ###sampling particles
xu = np.random.rand(N, T)  ####particles after resampling
q = np.random.rand(N, T)  ###normalised weights
qq = np.random.rand(N, T)  ###unnormalised weights
I = np.random.rand(N, T)  ###offsprings
R = np.random.rand(N, 1)  ###variance
#result = np.random.rand(T, 1)
x[:, 0] = np.random.normal(0, math.sqrt(initiVar), N)
for j in range(1, N + 1):
    R[j - 1] = (beta ** 2) * math.exp(x[j - 1, 0])
for j in range(1, N + 1):
    qq[j - 1, 0] = math.exp(-0.5 * (R[j - 1] ** (-1)) * (y[0] ** 2)) / math.sqrt(2 * math.pi * R[j - 1])
q[:, 0] = qq[:, 0] / math.fsum(qq[:, 0])
probs = q[:, 0].tolist()
max_weights = max(q[:, 0])
for j in range(1, N + 1):
    thre = np.random.uniform(0, 1)
    b = q[j - 1, 0] / max_weights
    if thre >= b:
        A = np.random.multinomial(1, probs)
        A = A.tolist()
        A = A.index(1)
        xu[j - 1, 0] = x[A, 0]
###update and prediction stages
for t in range(1, T):
    x[:, t] = rho * xu[:, t - 1] + sigma * np.random.normal(0, 1, N)
    for j in range(1, N + 1):
        R[j - 1] = (beta ** 2) * math.exp(x[j - 1, t])
    for j in range(1, N + 1):
        qq[j - 1, t] = math.exp(-0.5 * (R[j - 1] ** (-1)) * (y[t] ** 2)) / math.sqrt(2 * math.pi * R[j - 1])
    q[:, t] = qq[:, t] / sum(qq[:, t])
    ###resampling step
    probs = q[:, t].tolist()
    max_weights = max(q[:, t])
    for j in range(1, N + 1):
        thre = np.random.uniform(0, 1)
        b = q[j - 1, t] / max_weights
        if thre >= b:
            A = np.random.multinomial(1, probs)
            A = A.tolist()
            A = A.index(1)
            xu[j - 1, t] = x[A, t]
            xu[j - 1, 0:t] = xu[A, 0:t]
        else:
            xu[j - 1, t] = x[j - 1, t]

###FFBSm
W = np.random.rand(N, T)
den = np.random.rand(N, 1)
W[:, T - 1] = q[:, T - 1]
for n in range(1, T):
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            den[j - 1, 0] = np.sum(q[:, T - n - 1] * norm.pdf(x[j - 1, T - n], loc=rho * x[:, T - n - 1], scale=1))
        W[i - 1, T - n - 1] = q[i - 1, T - n - 1] * np.sum(W[:, T - n] * norm.pdf(x[:, T - n], loc=rho * x[i - 1, T - n - 1]) / den[:, 0])
app_x_smoothing = np.zeros([T, 1])
for t in range(1, T+1):
        app_x_smoothing[t-1] = np.sum(W[:, t-1]*x[:, t-1])


