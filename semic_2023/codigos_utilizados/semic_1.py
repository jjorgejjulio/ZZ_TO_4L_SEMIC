#Bibliotecas utilizadas
#Bibliotecas utilizadas
import pandas as pd
import vector
import numpy as np
import matplotlib.pyplot as plt  
import math
import csv
from scipy.optimize import curve_fit
import os
from scipy.stats import norm
import scipy.stats as stats
from scipy.stats import crystalball


pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')
#Arquivo csv contendo as informações de cada partícula (4 leptons)
pasta_csv = r"C:\Users\venut\Desktop\CSV"
# Lista para armazenar os caminhos completos de todos os arquivos CSV na pasta
arquivos_csv = []
#envia os arquivos da pasta para a lista criada
for arquivo in os.listdir(pasta_csv):
         arquivos_csv.append(os.path.join(pasta_csv, arquivo))
#junta os arquivos em 1 só
df = pd.concat((pd.read_csv(file_path) for file_path in arquivos_csv), axis=0)



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

#plt.figure(figsize=(6,6))
#plt.hist(x1data,label='Z1', bins=50)
#plt.legend()
#plt.show()

hist,bins=np.histogram(x3data,bins=48)
x1= (bins[:-1] + bins[1:]) / 2 
y=hist




beta = 50 #média da extensão da calda
#(média, variância, assimetria, curtose) com base nos seus dados
mean = np.mean(x3data)  
var = np.var(x3data)    
skewness = stats.skew(x3data)#assimetria
kurtosis = stats.kurtosis(x3data)#calda e achatamento##
sigma=np.std(x3data)

print(skewness)
print(kurtosis)


x = np.linspace(min(x3data), max(x3data),1000)


#                                 calda 0.56       18    centro 90.8       amplitude 3.99 
pdf_values = crystalball.pdf(0.97*x,   0.1,      18,    90.1,      1.1)










plt.plot(x, pdf_values, 'r-', lw=4, alpha=0.6, label='ajuste')

plt.hist(x3data, density=True, bins=60,label='Dados')

plt.xlim(40,110)
# Configurações do gráfico
plt.title('Distribuição de Cristal (Crystalball)')
plt.xlabel('Valores')
plt.ylabel('Densidade de Probabilidade')
plt.legend(loc='best', frameon=False)
plt.grid(True)

# Mostrar o gráfico
plt.show()