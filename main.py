import matplotlib.pyplot as plt
import numpy as np
import scipy
import xlwt

# book = xlwt.Workbook(encoding="utf-8")
#
# sheet1 = book.add_sheet("Python Sheet 1")
#
# cols = ["A"]
# txt = np.random.normal(0, np.sqrt(2.6), 40000)
#
# for i in range(40000):
#       sheet1.write(i, 0, txt[i])
#
# book.save("spreadsheet.xls")

EPS = np.array([-0.69803564, 0.171437979, -0.040201569,
0.227849812,
-0.114632133,
0.282282145,
0.764290000,
2.102680769,
-0.437563099,
1.002308259,
-0.946176469,
0.590997092,
1.140031622,
-0.747065139,
1.003187184,
2.023660848,
-0.383765703,
1.56819738,
0.717033404,
0.427626561,
-0.153636904,
-0.928055746,
-0.72936955,
-0.022506992,
1.629392066,
0.174047425,
0.479408342,
0.175466364,
-1.812102148,
-0.004225004,
-0.452657023,
0.833306372,
0.915799339,
0.166145205,
1.217391525,
0.693920425,
0.730873649,
-0.660745701,
-1.000377762,
-1.577939502])


x = np.linspace(-4 + 8/40, 4, 40)

theta = np.array([5.5, 5, -6, 0.07])
Y_real = np.zeros(40)
for j in range(0, 40):
    for i in range(0, theta.size):
        Y_real[j] += theta[i] * (x[j]**i)

Y = Y_real + EPS
print(Y)
# когда m = 1

X1T = np.array([np.ones(40), x])
X1 = np.transpose(X1T)

ans1 = np.dot(np.dot(np.linalg.inv(np.dot(X1T, X1)), X1T), Y)

sqrt_alph1 =np.sqrt(np.linalg.inv(np.dot(X1T, X1))[1][1])

E1 = Y - np.dot(X1, ans1)
mod_E1 = np.sqrt(np.dot(np.transpose(E1), E1))#(убрала корень)

T1 = (ans1[1] * np.sqrt(38))/(sqrt_alph1 * mod_E1)
#( ЧТОБЫ ВЗЯТЬ ДАННЫЕ ДЛЯ ОТЧЕТА ,НУЖНО ВОЗВЕСТИ ЗНАЧЕНИЯ В КВАДРАТ)
print(T1**2)
t1 = 2.0244

if t1 > abs(T1):
    print("Модель имеет порядок 0")
    m = 0
    theta_checked = ans1
    X = X1
    XT = X1T
    sqrt_alph = sqrt_alph1
    E = E1
    mod_E = mod_E1
else:
    # пусть m = 2

    X2T = np.array([np.ones(40), x, x*x])
    X2 = np.transpose(X2T)

    ans2 = np.dot(np.dot(np.linalg.inv(np.dot(X2T, X2)), X2T), Y)

    sqrt_alph2 = np.sqrt(np.linalg.inv(np.dot(X2T, X2))[2][2])

    E2 = Y - np.dot(X2, ans2)
    mod_E2 = np.sqrt(np.dot(np.transpose(E2), E2))

    T2 = (ans2[2] * np.sqrt(37))/(sqrt_alph2 * mod_E2)

    t2 = 2.0262

    if t2 > abs(T2):
        print("Модель имеет порядок 1")
        m = 1
        theta_checked = ans1
        X = X1
        XT = X1T
        sqrt_alph = sqrt_alph1
        E = E1
        mod_E = mod_E1
    else:
        # пусть m = 3

        X3T = np.array([np.ones(40), x, x*x, x*x*x])
        X3 = np.transpose(X3T)

        ans3 = np.dot(np.dot(np.linalg.inv(np.dot(X3T, X3)), X3T), Y)

        sqrt_alph3 = np.sqrt(np.linalg.inv(np.dot(X3T, X3))[3][3])

        E3 = Y - np.dot(X3, ans3)
        mod_E3 = np.sqrt(np.dot(np.transpose(E3), E3))

        T3 = (ans3[3] * np.sqrt(36))/(sqrt_alph3 * mod_E3)

        t3 = 2.0281


        print("")
        print("")
        print("")

        if t3 > abs(T3):
            print("Модель имеет порядок 2")
            print(ans2)

            print("")
            print("")
            print("")
            m = 2
            theta_checked = ans2
            X = X2
            XT = X2T
            sqrt_alph = sqrt_alph2
            E = E2
            mod_E = mod_E2
        else:
            # пусть m = 4

            X4T = np.array([np.ones(40), x, x*x, x*x*x, x*x*x*x])
            X4 = np.transpose(X4T)

            ans4 = np.dot(np.dot(np.linalg.inv(np.dot(X4T, X4)), X4T), Y)

            sqrt_alph4 = np.sqrt(np.linalg.inv(np.dot(X4T, X4))[4][4])

            E4 = Y - np.dot(X4, ans4)
            mod_E4 = np.sqrt(np.dot(np.transpose(E4), E4))

            T4 = (ans4[4] * np.sqrt(35))/(sqrt_alph4 * mod_E4)
            # print(sqrt_alph4)
            print("")
            # print(sqrt_alph3)
            t4 = 2.0301

            if t4 > abs(T4):
                print("Модель имеет порядок 3")
                m = 3
                theta_checked = ans3
                X = X3
                XT = X3T
                sqrt_alph = sqrt_alph3
                E = E3
                mod_E = mod_E3


Y_checked = np.zeros(40)
for j in range(0, 40):
    for i in range(0, theta_checked.size):
        Y_checked[j] += theta_checked[i] * (x[j]**i)

# номер 2

theta_intervals1 = np.array([np.ones(m+1), np.ones(m+1)]).transpose()
theta_intervals2 = np.array([np.ones(m+1), np.ones(m+1)]).transpose()


from scipy.stats import t

for i in range(0, m + 1):
    theta_intervals1[i][0] = -((mod_E * mod_E * sqrt_alph * t.ppf((1 + 0.95)/2, 40-m-1)) / np.sqrt(40-m-1)) + theta_checked[i]
    theta_intervals1[i][1] = ((mod_E * mod_E * sqrt_alph * t.ppf((1 + 0.95)/2, 40-m-1)) / np.sqrt(40-m-1)) + theta_checked[i]

# print(theta_intervals1)

for i in range(0, m + 1):
    theta_intervals2[i][0] = -(mod_E * np.sqrt(np.linalg.inv(np.dot(XT, X))[i][i]) * t.ppf((1 + 0.99)/2, 40-m-1)) / np.sqrt(40-m-1) + theta_checked[i]
    theta_intervals2[i][1] = (mod_E * np.sqrt(np.linalg.inv(np.dot(XT, X))[i][i]) * t.ppf((1 + 0.99)/2, 40-m-1)) / np.sqrt(40-m-1) + theta_checked[i]

print(theta_intervals2)

print("")
print("")
print("")
print("")

# номер 3

Y_right1 = np.zeros(40)
Y_left1 = np.zeros(40)
Y_right2 = np.zeros(40)
Y_left2 = np.zeros(40)


if m == 1:
    X_needed = np.array([np.ones(40), x])
if m == 2:
    X_needed = np.array([np.ones(40), x, x * x])
if m == 3:
    X_needed = np.array([np.ones(40), x, x * x, x * x * x])

X_needed = X_needed.transpose()


for i in range(0, 40):
    alph_temp = np.dot(np.dot(X_needed[i], np.linalg.inv(np.dot(XT, X))), X_needed[i])

    Y_left1[i] = -((mod_E * np.sqrt(alph_temp) * t.ppf((1 + 0.95)/2, 40-m-1)) / np.sqrt(40-m-1)) + Y_real[i]
    Y_right1[i] = ((mod_E * np.sqrt(alph_temp) * t.ppf((1 + 0.95)/2, 40-m-1)) / np.sqrt(40-m-1)) + Y_real[i]

    Y_left2[i] = -((mod_E * np.sqrt(alph_temp) * t.ppf((1 + 0.99) / 2, 40-m-1)) / np.sqrt(40-m-1)) + Y_real[i]
    Y_right2[i] = ((mod_E * np.sqrt(alph_temp) * t.ppf((1 + 0.99) / 2, 40-m-1)) / np.sqrt(40-m-1)) + Y_real[i]

Very_important = np.array([Y_left2, Y_right2]). transpose()

print(Very_important)
print("")
print("")
print("")

fig = plt.figure(figsize=[7, 5])
fig1 = plt.figure(figsize=[7, 5])
fig2 = plt.figure(figsize=[7, 5])

ax = fig.add_subplot(1, 1, 1)
ax1 = fig1.add_subplot(1, 1, 1)
ax2 = fig2.add_subplot(1, 1, 1)

ax.plot(x, Y_real, label = "real function")
ax.plot(x, Y, label = "function with noise", ls = 'None',marker='.')
ax.plot(x, Y_checked, label = "checked function")


EPS_max = max(E)
EPS_min = min(E)

step = (EPS_max - EPS_min) / 6

bars = []

for i in range (-1, 8):
    bars.append(EPS_min + step*i + 0.0001*i)
    print(EPS_min + step*i + 0.0001*i)



ax2.plot(x, Y_real, label = "real function")
# ax2.plot(x, Y_right1, label = "bigger function 0.95")
# ax2.plot(x, Y_left1, label = "lesser function 0.95")
ax2.plot(x, Y, label = "function with noise", ls = 'None',marker='.')
ax2.plot(x, Y_right2, label = "bigger function 0.99")
ax2.plot(x, Y_left2, label = "lesser function 0.99")

ax.legend()

ax2.legend()


n=40
# Оценили сигму квадрат с крышкой

sigma_hat_squared = (mod_E * mod_E/(n))
print(mod_E*mod_E)
print("")
print(sigma_hat_squared)
print("")
print(theta_checked)
print("")


# Проверили гипотезуо том,что EPS ~ N(0, sigma)

p_hat = np.zeros(8)
for i in range(0, 40):
    if E[i] >= bars[1] and E[i] <= bars[2]:
        p_hat[1] += 1

for i in range(0, 40):
    if E[i] > bars[2] and E[i] <= bars[3]:
        p_hat[2] += 1

for i in range(0, 40):
    if E[i] > bars[3] and E[i] <= bars[4]:
        p_hat[3] += 1

for i in range(0, 40):
    if E[i] > bars[4] and E[i] <= bars[5]:
        p_hat[4] += 1

for i in range(0, 40):
    if E[i] > bars[5] and E[i] <= bars[6]:
        p_hat[5] += 1
for i in range(0, 40):
    if E[i] > bars[6] and E[i] <= bars[7]:
        p_hat[6] += 1

for i in range(1, 7):

    p_hat[i] /= 40


p_not_hat = np.zeros(7)

p_not_hat[0] = scipy.stats.norm.cdf(EPS_min/np.sqrt(sigma_hat_squared)) - scipy.stats.norm.cdf(-100/np.sqrt(sigma_hat_squared))
p_not_hat[6] = scipy.stats.norm.cdf(100/np.sqrt(sigma_hat_squared)) - scipy.stats.norm.cdf(EPS_max/np.sqrt(sigma_hat_squared))

for i in range(1, 6):
    p_not_hat[i] = scipy.stats.norm.cdf(bars[i+1] / np.sqrt(sigma_hat_squared)) - scipy.stats.norm.cdf(bars[i]/np.sqrt(sigma_hat_squared))

TZn = 0
for i in range(0, 7):
    TZn += ((p_not_hat[i] - p_hat[i]) * (p_not_hat[i] - p_hat[i]))/p_not_hat[i]

TZn *= 40
print(TZn)
from scipy.stats import chi2

if  0 < TZn and TZn < chi2.ppf(0.95, 5):
    print("Распределение ошибок нормальное")
else:
    print("Не является нормальным")

ax1.hist(E, bars, density = True)


print(sigma_hat_squared)
print("")
print("")
print("")


# plt.show()
