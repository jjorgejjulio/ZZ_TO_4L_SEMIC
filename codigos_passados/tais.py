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

    if abs(row['PID1']) == abs(row['PID2']) and row['Q1'] != row['Q2']:
           return (v1+v2).mass

def compute_mass2(row):
    v1 = vector.obj(px=row['px1'], py=row['py1'], pz=row['pz1'], E=row['E1'])
    v2 = vector.obj(px=row['px2'], py=row['py2'], pz=row['pz2'], E=row['E2'])
    v3 = vector.obj(px=row['px3'], py=row['py3'], pz=row['pz3'], E=row['E3'])
    v4 = vector.obj(px=row['px4'], py=row['py4'], pz=row['pz4'], E=row['E4'])

    if abs(row['PID3']) == abs(row['PID4']) and row['Q3'] != row['Q4']:
           return (v3+v4).mass
    
df['massa1'] = df.apply(compute_mass1, axis=1)
df['massa2'] = df.apply(compute_mass2, axis=1)

x1data=np.array(df['massa1'])
x2data=np.array(df['massa2'])
x3data= np.hstack((x1data, x2data))



def compute_mass3(row):
    v1 = vector.obj(px=row['px1'], py=row['py1'], pz=row['pz1'], E=row['E1'])
    v2 = vector.obj(px=row['px2'], py=row['py2'], pz=row['pz2'], E=row['E2'])
    v3 = vector.obj(px=row['px3'], py=row['py3'], pz=row['pz3'], E=row['E3'])
    v4 = vector.obj(px=row['px4'], py=row['py4'], pz=row['pz4'], E=row['E4'])

    if abs(row['PID1']) == abs(row['PID3']) and row['Q1'] != row['Q3']:
           return (v1+v3).mass

def compute_mass4(row):
    v1 = vector.obj(px=row['px1'], py=row['py1'], pz=row['pz1'], E=row['E1'])
    v2 = vector.obj(px=row['px2'], py=row['py2'], pz=row['pz2'], E=row['E2'])
    v3 = vector.obj(px=row['px3'], py=row['py3'], pz=row['pz3'], E=row['E3'])
    v4 = vector.obj(px=row['px4'], py=row['py4'], pz=row['pz4'], E=row['E4'])

    if abs(row['PID2']) == abs(row['PID4']) and row['Q2'] != row['Q4']:
           return (v2+v4).mass
    
df['massa3'] = df.apply(compute_mass3, axis=1)
df['massa4'] = df.apply(compute_mass4, axis=1)

x4data=np.array(df['massa3'])
x5data=np.array(df['massa4'])
x6data= np.hstack((x4data, x5data))


def compute_mass5(row):
    v1 = vector.obj(px=row['px1'], py=row['py1'], pz=row['pz1'], E=row['E1'])
    v2 = vector.obj(px=row['px2'], py=row['py2'], pz=row['pz2'], E=row['E2'])
    v3 = vector.obj(px=row['px3'], py=row['py3'], pz=row['pz3'], E=row['E3'])
    v4 = vector.obj(px=row['px4'], py=row['py4'], pz=row['pz4'], E=row['E4'])

    if abs(row['PID1']) == abs(row['PID4']) and row['Q1'] != row['Q4']:
           return (v1+v4).mass

def compute_mass6(row):
    v1 = vector.obj(px=row['px1'], py=row['py1'], pz=row['pz1'], E=row['E1'])
    v2 = vector.obj(px=row['px2'], py=row['py2'], pz=row['pz2'], E=row['E2'])
    v3 = vector.obj(px=row['px3'], py=row['py3'], pz=row['pz3'], E=row['E3'])
    v4 = vector.obj(px=row['px4'], py=row['py4'], pz=row['pz4'], E=row['E4'])

    if abs(row['PID2']) == abs(row['PID3']) and row['Q2'] != row['Q3']:
           return (v2+v3).mass
    
df['massa5'] = df.apply(compute_mass5, axis=1)
df['massa6'] = df.apply(compute_mass6, axis=1)

x7data=np.array(df['massa5'])
x8data=np.array(df['massa6'])
x9data= np.hstack((x7data, x8data))

# passando os dados obtidos para um array para fazer o plot dos dados


#hist,bins=np.histogram(x6data,bins=38)
#x1= (bins[:-1] + bins[1:]) / 2 
#y=histhist,bins=np.histogram(x6data,bins=38)
#x1= (bins[:-1] + bins[1:]) / 2 
#y=hist

#def gaussian(x, A, mu, sigma):
#    return A * np.exp(-((x - mu)**2 )/ (2 * sigma**2))

# Ajuste de gaussiana
#params, covariance = curve_fit(gaussian, x1, y, p0=[max(y), np.mean(x1), np.std(x1)])  # Chute inicial p0=[A, mu, sigma]
#A_fit, mu_fit, sigma_fit = params

#x_fit = np.linspace(90,100, 500)  # Aumente o número de pontos

#                      amplitude, centro, abertura
#y_fit = gaussian(x_fit, A_fit*0.9, mu_fit, sigma_fit)





massa_corte = x6data
z_mass = 91.1876


x3data_lim = x3data[(x3data >= 50) & (x3data <= 120)]

# Calcular a média dos dados limitados
mean = np.mean(x3data_lim)
print(mean)
rms = np.sqrt(mean)
print(rms)

plt.figure(figsize=(8,8))
a=24
n, bins, patches = plt.hist(x3data, bins = a , range=(0,150),histtype='step', color = "#ffcc64")
n, bins, patches = plt.hist(x6data, bins = a , range=(0,150), histtype='step',color = "#e5ff64")
n, bins, patches = plt.hist(x9data, bins = a ,range=(0,150),histtype='step', color = "#ff7f64")
#plt.plot(x_fit, y_fit, color='green', label='Curva Gaussiana ajustada')


#patches[48].set_fc('#2596be')
plt.xlim(0,125)
#plt.ylim(0,3000)
plt.xlabel( '$m_{ℓ⁺ℓ⁻}$ [GeV]', loc="right", fontsize=35)
plt.ylabel("Eventos", loc="top",fontsize=35)

#blue_patch = mpatches.Patch(color='#2596be', label='Z → ℓ⁺ℓ⁻')
yellow_patch = mpatches.Patch(color='#ffcc64', label='Z → ℓ⁺ℓ⁻')

plt.legend(handles=[yellow_patch], edgecolor='white', title_fontsize= '50')
legenda_existente = plt.legend()


plt.legend(['P1P2-P3P4','P1P3-P2P4','P1P4-P2P3'])


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