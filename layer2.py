# See PyCharm help at https://www.jetbrains.com/help/pycharm/
from scipy.stats import invgamma
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

####RSMC with RMHMC moves
NR = 20 ###number of runs
MC = np.random.rand(NR, T)
for nr in range(1, NR+1):
    starttime = datetime.datetime.now()
    ###optimal proposal SI
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
    #result[0, 0] = np.sum(xu[:, 0]/N)
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
            #thre2 = math.log(T)/T
            #move_b = np.random.uniform(0, 1, 1)
            if t == 1 or t == 2 or t == 4 or t == 7 or t == 1 or t == 20 or t == 29:
                #print(t)
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
                    ppot = p + epsilon * grad(xu[j - 1, 0:(t + 1)]) / 2
                    xpot = xu[j - 1, 0:(t + 1)] + epsilon * G1.dot(ppot)
                    ppot = ppot + epsilon * grad(xpot) / 2
                    thre3 = math.exp(
                        -loglikeli(xu[j - 1, 0:(t + 1)]) + loglikeli(xpot) + 0.5 * p.dot(G1).dot(p) - 0.5 * ppot.dot(
                            G1).dot(ppot))
                    accept = np.random.uniform(0, 1)
                    if accept <= thre3:
                        xu[j - 1, 0:(t + 1)] = xpot
                        p = ppot
                    else:
                        xu[j - 1, 0:(t + 1)] = xu[j - 1, 0:(t + 1)]
                        p = p
    endtime = datetime.datetime.now()
    print(endtime - starttime)
        #result[t, 0] = np.sum(xu[:, t] / N)
    MC[nr - 1, :] = xu.mean(axis=0)
    for t in range(1, T+1):
        MC[nr-1, t-1] = np.sum(x[:, t-1]*q[:, t-1])###smoothing xu.mean(axis=0)

####RSMC with PRC
MC = np.random.rand(NR, T)
for nr in range(1, NR+1):
    starttime = datetime.datetime.now()
    c = np.random.rand(T, 1)###threhold of rejection
    x = np.random.rand(N, T)  ###sampling particles
    xu = np.random.rand(N, T)  ####particles after resampling
    xr = np.zeros([H, 1])  ####particles for rejection
    #xrr = np.random.rand(2*N, T)
    q = np.zeros([N, T])  ###normalised weights
    qq = np.random.rand(N, T)  ###unnormalised weights
    #qqq = np.random.rand(2*N, T) ###weights for rejection
    #I = np.random.rand(N, T)  ###offsprings
    R = np.random.rand(N, 1)  ###variance
    #var = np.random.rand(2*N, 1)  ###variance
    x[:, 0] = np.random.normal(0, math.sqrt(initiVar), N)
    for j in range(1, N + 1):
        R[j - 1] = (beta ** (2)) * math.exp(x[j - 1, 0])
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
    q[:, 0] = [1 / N] * N
    ###update and prediction stages
    for t in range(1, T):
        x[:, t] = rho * xu[:, t - 1] + sigma * np.random.normal(0, 1, N)
        for j in range(1, N + 1):
            qq[j - 1, t] = q[j-1, t-1]*norm.pdf(y[t], loc=0, scale=beta * math.exp(x[j-1, t]/2))
        c[t] = np.quantile(qq[np.absolute(qq[:, t]) != 0, t], .95)
        print("c[t, N]", c[t])
        c = np.nan_to_num(c, copy=True, nan=0.0, posinf=None, neginf=None)
        ####rejection control
        if c[t] == 0:
            x[:, t] = x[:, t]
            qq[:, t] = qq[:, t]
        else:
            i = 0
            while i < N:
                thre1 = np.random.uniform(0, 1)
                if thre1 <= (qq[i, t]/c[t]):
                    x[i, t] = x[i, t]
                    xr = [rho * xu[i, t - 1]]*H + sigma * norm.rvs(size=H)
                    for h in range(1, H+1):
                        xr[h-1] = norm.pdf(y[t], loc=0, scale=beta * math.exp(xr[h-1]/2))
                    xr = (q[i, t - 1] / c[t]) * xr
                    xr[np.absolute(xr) > 1] = 1
                    r = xr.mean()
                    qq[i, t] = qq[i, t]*r/min(1, qq[i, t]/c[t])
                    i = i + 1
                    print(i)
                else:
                    x[i, t] = rho * xu[i, t - 1] + sigma * norm.rvs(0, 1)
                    qq[i, t] = q[i, t - 1] * norm.pdf(y[t], loc=0, scale=beta * math.exp(x[i, t]/2))
        #q[:, t] = np.nan_to_num(qq[:, t], copy=True, nan=0.0, posinf=None, neginf=None)
        ###resampling step
        q[:, t] = qq[:, t] / sum(qq[:, t])
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
        q[:, t] = [1 / N] * N
    endtime = datetime.datetime.now()
    print(endtime - starttime)
    #for t in range(1, T+1):
    MC[nr-1, :] = xu.mean(axis=0)

####plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.suptitle('SV model: diversity')
xaxs = range(1, T + 1)
#app_x = np.zeros([T, N])
for i in range(1, N+1):
    ax4.plot(xaxs, xu[i-1, :], color="0.8")
ax4.plot(xaxs, true_x, "b-", label="true hidden state")
estimate_x = np.zeros([T, 1])
estimate_x = xu.mean(axis=0)
ax4.plot(xaxs, estimate_x, "r--", label="estimation")
ax4.set_title("PRC RSMC")

MC_1 = np.load('ESS_1_N50.npy')#('RMHMC_2_N50.npy')
MC_2 = np.load('DR_1_N60.npy')#('PRC_2_N60.npy')
MC_3 = np.load('RSMC_1_N50.npy')
MC_4 = np.load('ESS_layer1.npy')#('RMHMC_layer2.npy')
MC_5 = np.load('DR_1_N150.npy')#('PRC_RSMC_layer2.npy')
MC_6 = np.load('RSMC_layer1.npy')
MC_7 = np.load('ESS_1_N300.npy')#('RMHMC_2_N300.npy')
MC_8 = np.load('DR_1_N300.npy')#('PRC_2_N300.npy')
MC_9 = np.load('RSMC_1_N300.npy')
xaxs = range(1, T+1)
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)
fig.suptitle('SV model: smoothing estimation')
app_x_4 = MC_4.mean(axis=0)
for n in range(1, N+1):
    ax4.plot(xaxs, MC_4[n-1, :], color="0.8")
ax4.plot(xaxs, true_x, "g-", label="true hidden state")
ax4.plot(xaxs, app_x_4, "r--", label="mean estimation")
ax4.set_title("N=150, Adaptive Resampling")

app_x_5 = MC_5.mean(axis=0)
for n in range(1, N+1):
    ax5.plot(xaxs, MC_5[n-1, :], color="0.8")
ax5.plot(xaxs, true_x, "g-", label="true hidden state")
ax5.plot(xaxs, app_x_5, "r--", label="mean estimation")
ax5.set_title("N=150, Dynamic Resampling")

app_x_6 = MC_6.mean(axis=0)
for n in range(1, N+1):
    ax6.plot(xaxs, MC_6[n-1, :], color="0.8")
ax6.plot(xaxs, true_x, "g-", label="true hidden state")
ax6.plot(xaxs, app_x_6, "r--", label="mean estimation")
ax6.set_title("N=150, Rejection SMC")

asym_var_1 = np.zeros([T, 1])
asym_var_2 = np.zeros([T, 1])
asym_var_3 = np.zeros([T, 1])
MSE_1 = np.zeros([T, 1])
MSE_2 = np.zeros([T, 1])
MSE_3 = np.zeros([T, 1])

for t in range(1, T+1):
    asym_var_1[t-1, 0] = np.var(MC_7[:, t - 1])
    asym_var_2[t - 1, 0] = np.var(MC_8[:, t - 1])
    asym_var_3[t - 1, 0] = np.var(MC_9[:, t - 1])
    MSE_1[t-1, 0] = np.square(np.subtract([true_x[t - 1]] * NR, MC_1[:, t - 1])).mean()
    MSE_2[t - 1, 0] = np.square(np.subtract([true_x[t - 1]] * NR, MC_2[:, t - 1])).mean()
    MSE_3[t - 1, 0] = np.square(np.subtract([true_x[t - 1]] * NR, MC_3[:, t - 1])).mean()

plt.plot(xaxs, MSE_1, "k-", label="RSMC")
plt.plot(xaxs, MSE_2, "r-", label="PRC RSMC")
plt.plot(xaxs, MSE_4, "b-", label="RSMC with RMHMC moves")
fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3)
ax3.plot(xaxs, asym_var_1, "b-", label="Var: RSMC with RMHMC moves")
ax3.plot(xaxs, asym_var_2, "r-", label="Var: PRC RSMC")
ax3.plot(xaxs, asym_var_3, "k-", label="Var: RSMC")
#plt.xlabel('time t')
#plt.ylabel('variance')#('Var')
ax3.set_title('Variance of Particles(N=150)')
ax3.legend()
#ax1.show()


### aggregated in time
bias_1 = np.zeros([NR, 1])
bias_2 = np.zeros([NR, 1])
bias_3 = np.zeros([NR, 1])
t = 1
bias_1[0, 0] = (app_x_7[t-1] - true_x[t-1, 0])**2
bias_2[0, 0] = (app_x_8[t-1] - true_x[t-1, 0])**2
bias_3[0, 0] = (app_x_9[t-1] - true_x[t-1, 0])**2

for t in range(2, T+1):
    bias_1[t-1, 0] = bias_1[t-2, 0] + (app_x_7[t-1] - true_x[t-1, 0])**2
    #bias_2[t-1, 0] = bias_2[t-2, 0] + (app_x_5[t-1] - true_x[t-1, 0])**2
    bias_3[t-1, 0] = bias_3[t-2, 0] + (app_x_9[t-1] - true_x[t-1, 0])**2
for t in range(1, T+1):
    bias_1[t-1, 0] = bias_1[t-1, 0]/t
    #bias_2[t-1, 0] = bias_2[t-1, 0]/t
    bias_3[t-1, 0] = bias_3[t-1, 0]/t

for nr in range(1, NR+1):
    bias_1[nr-1] = sum((MC_7[nr-1, :] - true_x[:, 0])**2)/T
    bias_2[nr-1] = sum((MC_8[nr-1, :] - true_x[:, 0])**2)/T
    bias_3[nr-1] = sum((MC_9[nr-1, :] - true_x[:, 0])**2)/T

plt.plot(xaxs, bias_1, "b-", label="RSMC with RMHMC moves")
plt.plot(xaxs, bias_2, "r-", label="PRC RSMC")
#plt.plot(xaxs, bias_3, "g-", label="PRC")
plt.plot(xaxs, bias_4, "k-", label="RSMC")
plt.xlabel('time t')
plt.ylabel('MSE')
plt.title('MSE for the new estimator')
plt.legend()
plt.show()


for t in range(1, T+1):
    app_x[t-1] = np.sum(xu[:, t-1]*q[:, T-1])
sd = np.zeros([T, 1])
for t in range(1, T+1):
    sd[t-1] = (np.sum((xu[:, t-1]**2)*q[:, T-1]) - app_x[t-1]**2)**(1/2)
axs[1, 0].plot(xaxs, app_x, 'r-', label="Mean")
axs[1, 0].plot(xaxs, app_x+sd, 'r--', label="+1 S.D.")
axs[1, 0].plot(xaxs, app_x-sd, 'r--', label="-1 S.D.")

axs[1, 0].plot(xaxs, true_x, 'b-', label="true trajectory")
axs[1, 0].set_title('RSMC')
plt.xlabel('time t')
plt.ylabel('hidden state x')
#plt.title('true trajectory and approximation of RSMC')
plt.title('SV Model: Rejection SMC with HMC moves for Smoothing')
plt.legend()
plt.show()

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
fig.suptitle(' Empirical distributions of the particle weights')
ax1.hist(qq[:, 29]/N)
ax1.set_title('N=150, RSMC')
ax1.set_xlim([0, 0.025])

ax4.hist(qq[:, 29])
ax4.set_title('N=60, RSMC with PRC')
ax4.set_xlim([0, 0.025])

ax3.hist(qq[:, 29]/sum(qq[:, 29]))
ax3.set_title('N=300,RSMC')


xaxs = range(1, T + 1)
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
fig.suptitle(' Smoothing estimates and the diversity for the SV model.')
for i in range(1, N+1):
    ax1.plot(xu[i-1, :], color='0.8')
app_x = np.zeros([T, 1])
for t in range(1, T+1):
    app_x[t-1] = np.sum(xu[:, t-1]/N)
ax1.plot(xaxs, app_x, 'r--', label="Mean")
ax1.plot(xaxs, true_x, 'g-', label="True trajectory")
ax1.set_title('RSMC, N=60')
# with RMHMC moves

sd = np.zeros([T, 1])
for t in range(1, T+1):
    sd[t-1] = (np.sum((xu[:, t-1]**2)*q[:, T-1]) - app_x[t-1]**2)**(1/2)
ax7.plot(xaxs, app_x+sd, 'r--', label="+1 S.D.")
ax7.plot(xaxs, app_x-sd, 'r--', label="-1 S.D.")
