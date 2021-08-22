import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import invgamma

# np.random.seed(189)

N = 150  ###number of samples or 300
T = 30  ###number of time steps
rho: float = 0.95 # 0.94
sigma = 1 #0.1
beta: float = 0.5 #0.98
initiVar = (sigma ** 2) / (1 - rho ** 2)  ##initial variance
#true_x = np.load('true_x_PMCMC.npy')
y = np.load('y_gradient.npy')
x0 = np.random.normal(0, math.sqrt(initiVar), 1)
true_x = np.random.rand(T, 1)  ###hidden states
y = np.random.rand(T, 1)
true_x[0] = x0
y[0] = beta * math.exp(true_x[0] / 2) * np.random.normal(0, 1, 1)

###true tracjatory
for t in range(1, T):
    true_x[t] = rho * true_x[t - 1] + sigma * np.random.normal(0, 1, 1)
    y[t] = beta * math.exp(true_x[t - 1] / 2) * np.random.normal(0, 1, 1)


def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

def grad(x, theta):
    beta = theta[0]
    sigma = theta[1]
    rho = theta[2]

    l: int = len(x)
    s = np.arange(l)
    r = np.arange(l)
    for i in range(1, l + 1):
        s[i - 1] = ((y[i - 1] ** 2) * math.exp(-x[i - 1]) / (beta ** 2) - 1) * 0.5
    r[0] = (x[0] - rho * x[1]) / (sigma ** 2)
    r[l - 1] = (x[l - 1] - rho * x[l - 2]) / (sigma ** 2)
    for i in range(1, l - 1):
        r[i] = (-rho * x[i + 1] + (1 + rho ** 2) * x[i] - rho * x[i - 1]) / (sigma ** 2)
    result = np.subtract(s, r)
    return result

def loglikeli(x, theta):
    beta = theta[0]
    sigma = theta[1]
    rho = theta[2]
    l = len(x)
    prod = math.log(
        norm.pdf(x[0], loc=0, scale=math.sqrt(initiVar)) * norm.pdf(y[0], loc=0, scale=beta * math.exp(x[0] / 2)))
    for i in range(1, l):
        prod = prod + math.log(norm.pdf(x[i], loc=rho * x[i - 1], scale=sigma) * norm.pdf(y[i], loc=0,
                                                                                          scale=beta * math.exp(
                                                                                              x[i] / 2)))
    result = prod
    return result

def gradient(theta):
    beta: float = theta[0]
    sigma = theta[1]
    rho: float = theta[2] #math.tanh(theta[2])  ###phi
    theta = [beta, sigma, rho]
    x = np.random.rand(N, T)  ###sampling particles
    xu = np.random.rand(N, T)  ####particles after resampling
    q = np.random.rand(N, T)  ###normalised weights
    qq = np.random.rand(N, T)  ###unnormalised weights
    R = np.random.rand(N, 1)  ###variance
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
            ###RMHMC
            # thre2 = math.log(T) / T
            # move_b = np.random.uniform(0, 1, 1)
            if t == 1 or t == 4 or t == 11 or t == 20 or t == 29:
                epsilon = 0.1  ### or 0.05
                diag = [(1 + rho ** 2) / (sigma ** 2)] * (t + 1)
                diag[0] = 1 / (sigma ** 2)
                diag[t] = 1 / (sigma ** 2)
                sub = [-rho / (sigma ** 2)] * t
                up = [-rho / (sigma ** 2)] * t
                G = 0.5 * np.identity(t + 1) + tridiag(sub, diag, up)
                G1 = np.linalg.inv(G)
                mean = [0] * (t + 1)
                p = np.random.multivariate_normal(mean=mean, cov=G1)
                for k in range(1, 6):
                    ppot = p + 0.5 * epsilon * grad(xu[j - 1, 0:(t + 1)], theta=theta)
                    xpot = xu[j - 1, 0:(t + 1)] + epsilon * G1.dot(ppot)
                    ppot = ppot + 0.5 * epsilon * grad(xpot, theta=theta)
                    thre3 = math.exp(
                        -loglikeli(xu[j - 1, 0:(t + 1)], theta=theta) + loglikeli(xpot, theta=theta) + 0.5 * p.dot(
                            G1).dot(p) - 0.5 * ppot.dot(
                            G1).dot(ppot))
                    accept = np.random.uniform(0, 1)
                    if accept <= thre3:
                        xu[j - 1, 0:(t + 1)] = xpot
                        p = ppot
                    else:
                        xu[j - 1, 0:(t + 1)] = xu[j - 1, 0:(t + 1)]
                        p = p

    result = np.zeros([N, 1])
    result[:, 0] = (1 - rho ** 2) * rho * (1 + xu[:, 0] ** 2) / (sigma ** 2)
    for t in range(1, T):
        result[:, 0] = result[:, 0] + xu[:, t - 1] * (xu[:, t] - rho * xu[:, t - 1]) / (sigma ** 2)
    result[:, 0] = result[:, 0]
    result = result.mean()
    return result


likelihood = np.zeros([100, 1])
for k in range(1, 101):
    likelihood[k - 1] = gradient([0.5, 1, k / 100])
haxs = np.array(range(1, 100+1))/100
plt.plot(haxs, np.load('likelihood_RMHMC_30.npy'), 'b-', label='RSMC with RMHMC moves')
plt.plot(haxs, np.load('likelihood_RSMC_30.npy'), 'g-', label='RSMC')
plt.plot(haxs, np.load('likelihood_ESS.npy'), 'r-', label='adaptive resampling')
plt.plot(haxs, [0] * len(haxs), 'k-')
plt.axvline(x=0.95, color='0.6')
plt.xlabel('phi')
plt.ylabel('gradient')  # ('Var')
plt.title('Function of gradient')  # ('Empirical Variance')
plt.legend()
plt.show()
###for 0.95
K = 500
phi = np.zeros([K, 1])
phi[0, 0] = 0.5
G = np.zeros([K, 1])
G[1, 0] = gradient([0.5, 1, phi[0, 0]])
phi_pot = phi[0, 0] + 0.005 * G[1, 0]
if phi_pot >= 1 or phi_pot <= -1:
    phi[1, 0] = phi[0, 0]
    a = 0
else:
    phi[1, 0] = phi_pot
    a = 1
for k in range(6, K):
    if a == 0:
        G[k, 0] = G[k-1, 0]
        phi_pot = phi[k - 1, 0] + G[k, 0]/(k+350)
        if 1 >= phi_pot >= -1:
            phi[k, 0] = phi_pot
            a = 1
        else:
            phi[k, 0] = phi[k - 1, 0]
            a = 0
    else:
        G[k, 0] = gradient([0.5, 1, phi[k - 1, 0]])
        phi_pot = phi[k - 1, 0] + G[k, 0]/(k+350)
        if phi_pot >= 1 or phi_pot <= -1:
            phi[k, 0] = phi[k - 1, 0]
            a = 0
        else:
            phi[k, 0] = phi_pot
            a = 1
#abs((phi[k - 1] - phi[k - 2]) / (G[k - 1] - G[k - 2])) * gradient([0.94, 0.1, phi[k - 1]])
    #G[k] = gradient([0.94, 0.1, phi[k]])

haxs = np.array(range(1, 500+1))
plt.plot(haxs, np.load('phi_RMHMC_0.95.npy'), 'b-', label='RSMC with RMHMC moves')
plt.plot(haxs, np.load('phi_RSMC_0.95.npy'), 'g-', label='RSMC')
plt.plot(haxs, np.load('phi_ESS_0.95.npy'), 'r-', label='adaptive resampling')
plt.plot(haxs, [0.95] * len(haxs), 'k-')
#plt.axvline(x=0.95, color='0.6')
plt.xlabel('iteration')
plt.ylabel('phi')  # ('Var')
plt.title('Estimation for static parameter obtained from gradient descent method')  # ('Empirical Variance')
plt.legend()
plt.show()



