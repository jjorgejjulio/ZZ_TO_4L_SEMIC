# Bibliotecas utilizadas
import pandas as pd
import vector
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
from scipy.optimize import curve_fit
import os

import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

# Arquivo csv contendo as informações de cada partícula (4 leptons)
pasta_csv = r"C:\Users\venut\Desktop\CSV"
# Lista para armazenar os caminhos completos de todos os arquivos CSV na pasta
arquivos_csv = []
# envia os arquivos da pasta para a lista criada
for arquivo in os.listdir(pasta_csv):
    arquivos_csv.append(os.path.join(pasta_csv, arquivo))
# junta os arquivos em 1 só
df = pd.concat((pd.read_csv(file_path) for file_path in arquivos_csv), axis=0)


# Definição da função que faz as condições de combinação e encontra a massa invariante de Z1
def compute_mass1(row):
    v1 = vector.obj(px=row['px1'], py=row['py1'], pz=row['pz1'], E=row['E1'])
    v2 = vector.obj(px=row['px2'], py=row['py2'], pz=row['pz2'], E=row['E2'])
    v3 = vector.obj(px=row['px3'], py=row['py3'], pz=row['pz3'], E=row['E3'])
    v4 = vector.obj(px=row['px4'], py=row['py4'], pz=row['pz4'], E=row['E4'])
    ideal = row['mZ1']
    a = abs(ideal - (v1 + v2).mass)
    b = abs(ideal - (v1 + v3).mass)
    c = abs(ideal - (v1 + v4).mass)
    lista1 = [a, b]
    lista2 = [a, c]
    lista3 = [b, c]
    menor1 = min(lista1)
    menor2 = min(lista2)
    menor3 = min(lista3)

    if abs(row['PID1']) == abs(row['PID3']) and abs(row['PID1']) == abs(row['PID4']) and row['Q1'] == row['Q2']:
        if b == menor3 and (v1 + v3).mass < 110:
            return (v1 + v3).mass
        else:
            return (v1 + v4).mass

    if abs(row['PID1']) == abs(row['PID2']) and abs(row['PID1']) == abs(row['PID3']) and row['Q1'] != row['Q2'] and row['Q1'] == row['Q4']:
        if a == menor1 and (v1 + v2).mass < 110:
            return (v1 + v2).mass
        else:
            return (v1 + v3).mass

    if abs(row['PID1']) == abs(row['PID2']) and abs(row['PID1']) == abs(row['PID4']) and row['Q1'] != row['Q2'] and row['Q1'] == row['Q3']:
        if a == menor2 and (v1 + v2).mass < 110:
            return (v1 + v2).mass
        else:
            return (v1 + v4).mass

    if row['Q1'] != row['Q2'] and abs(row['PID1']) == abs(row['PID2']) and abs(row['PID1']) != abs(row['PID3']):
        return (v1 + v2).mass


df['massa1'] = df.apply(compute_mass1, axis=1)

# Definição da função que faz as condições de combinação e encontra a massa invariante de Z2
def compute_mass2(row):
    v1 = vector.obj(px=row['px1'], py=row['py1'], pz=row['pz1'], E=row['E1'])
    v2 = vector.obj(px=row['px2'], py=row['py2'], pz=row['pz2'], E=row['E2'])
    v3 = vector.obj(px=row['px3'], py=row['py3'], pz=row['pz3'], E=row['E3'])
    v4 = vector.obj(px=row['px4'], py=row['py4'], pz=row['pz4'], E=row['E4'])
    ideal = row['mZ2']
    a = abs(ideal - (v3 + v4).mass)
    b = abs(ideal - (v2 + v3).mass)
    c = abs(ideal - (v2 + v4).mass)
    lista1 = [a, b]
    lista2 = [a, c]
    lista3 = [b, c]
    menor1 = min(lista1)
    menor2 = min(lista2)
    menor3 = min(lista3)

    if abs(row['PID3']) == abs(row['PID4']) and abs(row['PID3']) == abs(row['PID2']) and row['Q3'] == row['Q4']:
        if b == menor3 and (v3 + v2).mass < 110:
            return (v3 + v2).mass
        else:
            return (v4 + v2).mass

    if abs(row['PID3']) == abs(row['PID4']) and abs(row['PID4']) == abs(row['PID2']) and row['Q3'] != row['Q4'] and row['Q3'] == row['Q2']:
        if a == menor2 and (v3 + v4).mass < 110:
            return (v3 + v4).mass
        else:
            return (v4 + v2).mass

    if abs(row['PID3']) == abs(row['PID4']) and abs(row['PID3']) == abs(row['PID2']) and row['Q3'] != row['Q4'] and row['Q4'] == row['Q2']:
        if a == menor1 and (v3 + v4).mass < 110:
            return (v3 + v4).mass
        else:
            return (v3 + v2).mass

    if row['Q3'] != row['Q4'] and abs(row['PID3']) == abs(row['PID4']) and abs(row['PID3']) != abs(row['PID2']):
        return (v3 + v4).mass


df['massa2'] = df.apply(compute_mass2, axis=1)

# passando os dados obtidos para um array para fazer o plot dos dados
x1data = np.array(df['massa1'])
x2data = np.array(df['massa2'])
x3data = np.hstack((x1data, x2data))
x4data = np.array(df['mZ1'])
x5data = np.array(df['mZ2'])
x6data = np.hstack((x4data, x5data))

hist,bins=np.histogram(x3data,bins=38)
x1= (bins[:-1] + bins[1:]) / 2 
y=histhist,bins=np.histogram(x3data,bins=38)

x1= (bins[:-1] + bins[1:]) / 2 
y=hist


def gaussian(x, A, mu, sigma):
    return A * np.exp(-((x - mu)**2 )/ (2 * sigma**2))
# Ajuste de gaussiana
params, covariance = curve_fit(gaussian, x1, y, p0=[max(y), np.mean(x1), np.std(x1)])  # Chute inicial p0=[A, mu, sigma]
A_fit, mu_fit, sigma_fit = params

x_fit1 = np.linspace(90,100, 500)  # Aumente o número de pontos
#                      amplitude, centro, abertura
y_fit1 = gaussian(x_fit1, A_fit*0.9, mu_fit, sigma_fit)

print(np.diag(covariance))
print(params)

def exponential_func(x, a, b, c):
    return a * np.exp(b * (x - 80)) + c

# Filtrar os dados para o intervalo de 80 a 90
mask = (x1 >= 80) & (x1 <= 90.2)
x_filtered = x1[mask]
y_filtered = y[mask]

# Realizar o ajuste da função exponencial
params, covariance = curve_fit(exponential_func, x_filtered, y_filtered)

# Parâmetros ajustados
a, b, c = params

# Gerar pontos para a curva ajustada
x_fit2 = np.linspace(80, 90, 100)
y_fit2 = exponential_func(x_fit2, a, b, c)

print(a)
print(b)
print(c)




massa_corte = x6data
z_mass = 91.1876
x3data_lim = x3data[(x3data >= 80) & (x3data <= 100)]
# Calcular a média dos dados limitados
mean = np.mean(x3data_lim)
rms = np.sqrt(mean)


plt.figure(figsize=(8,8))
n, bins, patches = plt.hist(massa_corte, bins = 50 , color = "#ffcc64")
plt.plot(x_fit1, y_fit1, color='green', label='Curva Gaussiana ajustada')
plt.plot(x_fit2, y_fit2, color='red', label='Curva exponencial ajustada')

plt.axvline(mean, color='red', linestyle='dashed', linewidth=1)
plt.axvline(rms, color='white', linestyle='dashdot', linewidth=1)
plt.xlim(80,100)
plt.xlabel( '$m_{ℓ⁺ℓ⁻}$ [GeV]', loc="right", fontsize=35)
plt.ylabel("Events", loc="top",fontsize=35)


yellow_patch = mpatches.Patch(color='#ffcc64', label='Z → ℓ⁺ℓ⁻')

plt.legend(handles=[yellow_patch], edgecolor='white', title_fontsize= '50')
legenda_existente = plt.legend()


nova_legenda = plt.legend([patches[0], plt.axvline(mean, color='r', linestyle='dashed', linewidth=1),
                           plt.axvline(rms, color='white', linestyle='dashdot', linewidth=1),
                           plt.axvline(z_mass, color='#2596be', linestyle='dashdot', linewidth=1)],
                            ['ℓ⁺ℓ⁻','mean = 90.6513','RMS = 9.5211',  'z mass = 91.1876±0.0021'], edgecolor='white', prop={'size':13})


plt.title("CMS", loc="left", fontweight='bold',fontsize=25)
plt.title("2011, 7 TeV and 2012, 8 Tev", loc="right",fontsize=25)

plt.tick_params(axis='both', bottom=True, top=True, left=True, right=True,
                direction='in', length=6, width=1, colors='black',labelsize=23)
plt.tick_params(axis='both', which='minor', direction='in', length=3, width=1, colors='black', top=True, right=True,labelsize=25)


ax = plt.gca()
ax.set_adjustable('datalim')
ax.xaxis.set_minor_locator(MultipleLocator(0.625))
ax.yaxis.set_minor_locator(MultipleLocator(62.50))

plt.show()