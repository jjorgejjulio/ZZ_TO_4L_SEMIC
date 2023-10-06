#Bibliotecas utilizadas
import vector
import numpy as np
import awkward as ak
import numba as nb
import numpy as np#library, it adds support for large, multi-dimensional arrays and matrices, it contains mathematical functions to operate on these arrays
import pandas as pd #numpy is a prerequisite. It organizes data and manipulate the data by putting it in a tabular form
import matplotlib.pyplot as plt #It creates a figure, creates a plotting area in a figure, plots some lines in a plotting area, decorates the plot with labels
import math
import csv
import scipy.stats as stats
from scipy.optimize import curve_fit

#Arquivo csv contendo as informações de cada partícula (4 leptons)
csvs = [pd.read_csv('https://raw.githubusercontent.com/GuillermoFidalgo/Python-for-STEM-Teachers-Workshop/master/data/4e_2011.csv'),
        pd.read_csv('https://raw.githubusercontent.com/GuillermoFidalgo/Python-for-STEM-Teachers-Workshop/master/data/4mu_2011.csv'),
        pd.read_csv('https://raw.githubusercontent.com/GuillermoFidalgo/Python-for-STEM-Teachers-Workshop/master/data/2e2mu_2011.csv')]
csvs += [pd.read_csv('https://raw.githubusercontent.com/GuillermoFidalgo/Python-for-STEM-Teachers-Workshop/master/data/4mu_2012.csv'), 
         pd.read_csv('https://raw.githubusercontent.com/GuillermoFidalgo/Python-for-STEM-Teachers-Workshop/master/data/4e_2012.csv'), 
         pd.read_csv('https://raw.githubusercontent.com/GuillermoFidalgo/Python-for-STEM-Teachers-Workshop/master/data/2e2mu_2012.csv')]
df = pd.concat(csvs)


#definição da primeira função que determina a massa invariante da partícula que emitiu um par de lepton e anti-lepton
def compute_mass1(row):
    v1 =vector.obj(px=row['px1'], py=row['py1'], pz=row['pz1'], E=row['E1'])
    v2 =vector.obj(px=row['px2'], py=row['py2'], pz=row['pz2'], E=row['E2'])
    v3 =vector.obj(px=row['px3'], py=row['py3'], pz=row['pz3'], E=row['E3'])
    v4 =vector.obj(px=row['px4'], py=row['py4'], pz=row['pz4'], E=row['E4'])
    if (abs(row['PID1'])==abs(row['PID2']) and (row['Q1']!=row['Q2'])) and (v1 + v2).mass < 109.5:
       return (v1 + v2).mass
    elif (abs(row['PID1'])==abs(row['PID3']) and (row['Q1']!=row['Q3'])) and (v1 + v3).mass < 109.5: 
       return (v1 + v3).mass
    elif (abs(row['PID1'])==abs(row['PID4']) and (row['Q1']!=row['Q4'])) and (v1 + v4).mass < 109.5: 
       return (v1 + v4).mass
df['massa1'] = df.apply(compute_mass1, axis=1)

#definição da segunda função que determina a massa invariante da partícula que emitiu um par de lepton e anti-lepton
def compute_mass2(row):
    v1 =vector.obj(px=row['px1'], py=row['py1'], pz=row['pz1'], E=row['E1'])
    v2 =vector.obj(px=row['px2'], py=row['py2'], pz=row['pz2'], E=row['E2'])
    v3 =vector.obj(px=row['px3'], py=row['py3'], pz=row['pz3'], E=row['E3'])
    v4 =vector.obj(px=row['px4'], py=row['py4'], pz=row['pz4'], E=row['E4'])
    if (abs(row['PID3'])==abs(row['PID4']) and (row['Q3']!=row['Q4'])) and (v3 + v4).mass < 109.5:
       return (v3 + v4).mass
    elif (abs(row['PID3'])==abs(row['PID2']) and (row['Q3']!=row['Q2'])) and (v3 + v2).mass < 109.5: 
       return (v3 + v2).mass
    elif (abs(row['PID4'])==abs(row['PID2']) and (row['Q4']!=row['Q2'])) and (v4 + v2).mass < 109.5: 
       return (v4 + v2).mass
df['massa2'] = df.apply(compute_mass2, axis=1)

#passando os dados obtidos para um array para fazer o plot dos dados
x1data=np.array(df['massa1'])
x2data=np.array(df['massa2'])
x3data=np.concatenate([x1data,x2data])
xref1=np.array(df['mZ1'])
xref2=np.array(df['mZ2'])
xref3=np.concatenate([xref1,xref2])




#fit para o primeiro plot
def gaussian(x, A, mu,sig):
    return A*np.exp(-(x-mu)**2/sig**2)
#Primeiro fit
hist,bins=np.histogram(x1data,bins=50)
x1= (bins[:-1] + bins[1:]) / 2
y1=hist
p01=[50,x1data.mean(),np.std(x1data)]
popt1, pcov1 = curve_fit(gaussian, x1, y1, p0=p01)
y_fit1 = gaussian(x1, *popt1)

#plot dos dados da primeira massa reconstruida
plt.figure(figsize=(6,6))
plt.hist(x1data,label='Z1', bins=50)
plt.title('$ \sqrt{s} = 7$ TeV, L = 2.3 $fb^{-1}$; $\sqrt{s} = 8$ TeV, L = 11.6 $fb^{-1}$ \n', fontsize = 12)
plt.xlabel('Massa invariante (GeV)', fontsize = 15)
plt.ylabel('Frequência', fontsize = 15)
#FIT
plt.plot(x1, y_fit1, 'r-', label='fit1')

plt.legend()
plt.show()




#Segundo Fit
hist,bins=np.histogram(x2data,bins=50)
x3=(bins[:-1] + bins[1:]) / 2 
y3=hist*1.4*0.8
p03=[45,89,np.std(x2data)]
popt3, pcov3 = curve_fit(gaussian,x3,y3,p0=p03)
y_fit3 = gaussian(x3, *popt3)
#Segundo Histograma
plt.figure(figsize=(6,6))
plt.hist(x2data,label='Z2',color='darkblue',bins=50)
#FIT
plt.plot(x3, y_fit3,'lightgreen',label='fit2')

plt.title('$ \sqrt{s} = 7$ TeV, L = 2.3 $fb^{-1}$; $\sqrt{s} = 8$ TeV, L = 11.6 $fb^{-1}$ \n', fontsize = 12)
plt.xlabel('Massa invariante (GeV)', fontsize = 15)
plt.ylabel('Frequência', fontsize = 15)
plt.legend()
plt.show()


#junção dos plots
#FIT4
hist,bins=np.histogram(x3data,bins=35)
x4= (bins[:-1] + bins[1:]) / 2
y4=hist
p04=[200,x3data.mean(),np.std(x3data)]
popt4, pcov4 = curve_fit(gaussian, x4, y4, p0=p04)
y_fit4 = gaussian(x4, *popt4)




#Plot dos histogramas juntos
plt.figure(figsize=(6,6))
plt.hist(x3data,label='Z1', bins=35)
plt.title('$ \sqrt{s} = 7$ TeV, L = 2.3 $fb^{-1}$; $\sqrt{s} = 8$ TeV, L = 11.6 $fb^{-1}$ \n', fontsize = 12)
plt.xlabel('Massa invariante (GeV)', fontsize = 15)
plt.ylabel('Frequência', fontsize = 15)
plt.legend()
#Fit
plt.plot(x4, y_fit4, 'r-', label='fit4')

plt.title('$ \sqrt{s} = 7$ TeV, L = 2.3 $fb^{-1}$; $\sqrt{s} = 8$ TeV, L = 11.6 $fb^{-1}$ \n', fontsize = 12)
plt.xlabel('Massa invariante (GeV)', fontsize = 15)
plt.ylabel('Frequência', fontsize = 15)
plt.legend()
plt.show()


# histograma de referencia (mZ1 e mZ2)
plt.figure(figsize=(6,6))
plt.hist(xref3,label='Z1', bins=35)
plt.title('referência,$ \sqrt{s} = 7$ TeV, L = 2.3 $fb^{-1}$; $\sqrt{s} = 8$ TeV, L = 11.6 $fb^{-1}$ \n', fontsize = 12)
plt.xlabel('Massa invariante (GeV)', fontsize = 15)
plt.ylabel('Frequência', fontsize = 15)


plt.legend()
plt.show()


#comparação entre os plots
plt.figure(figsize=(6, 6))
plt.hist(xref3, label='Referência', bins=35, alpha=0.5)  # Histograma de referência (parcialmente transparente)
plt.hist(x3data, label='Calculado', bins=35,alpha=0.3)  # Histograma de x3data
plt.title('$ \sqrt{s} = 7$ TeV, L = 2.3 $fb^{-1}$; $\sqrt{s} = 8$ TeV, L = 11.6 $fb^{-1}$ \n', fontsize=12)
plt.xlabel('Massa invariante (GeV)', fontsize=15)
plt.ylabel('Frequência', fontsize=15)
plt.legend()
plt.show()



