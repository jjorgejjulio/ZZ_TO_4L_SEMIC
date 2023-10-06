#Bibliotecas utilizadas
import pandas as pd
import vector
import numpy as np
import matplotlib.pyplot as plt  
import math
import csv
from scipy.optimize import curve_fit
import os

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

x10data=np.array(df['mZ1'])
x11data=np.array(df['mZ2'])
x12data= np.hstack((x10data, x11data))


a=100
b=5
#plots
#plt.hist(x3data, bins=a, width=b, label='P1P2 -- P3P4', color='darkblue')
#plt.hist(x6data, bins=a, width=b, label='P1P3 -- P2P4', color='darkorange')
#plt.hist(x9data, bins=a, width=b, label='P1P4 -- P2P3', color='darkgreen')
#plt.hist(x12data,bins=a, alpha=0.4, width=b, label='ZZ', color='darkred')
#plt.xlim(50,140)


#legendas
plt.title('$ \sqrt{s} = 7$ TeV, L = 2.3 $fb^{-1}$; $\sqrt{s} = 8$ TeV, L = 11.6 $fb^{-1}$ \n', fontsize=30)
plt.xlabel('Massa invariante (GeV/$c^{2}$)', fontsize=30)
plt.ylabel('Eventos', fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

legenda = plt.legend(fontsize=25)
for texto_legenda in legenda.get_texts():
    texto_legenda.set_fontsize(25)



#plt.text(0.88, 1.04, "(Z → ℓ⁺ℓ⁻)", transform=plt.gca().transAxes, fontsize=25, ha='left')
#plt.show()
print('a')


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
x10data=np.array(df['mZ1'])
x11data=np.array(df['mZ2'])
x12data= np.hstack((x10data, x11data))

plt.hist(x3data, bins=a, width=b, label='P1P2 -- P3P4', color='blue')
#plt.hist(x12data,bins=a,  width=b, label='ZZ', color='darkred')
plt.xlim(50,140)



#legendas
plt.title('$ \sqrt{s} = 7$ TeV, L = 2.3 $fb^{-1}$; $\sqrt{s} = 8$ TeV, L = 11.6 $fb^{-1}$ \n', fontsize=30)
plt.xlabel('Massa invariante (GeV/$c^{2}$)', fontsize=30)
plt.ylabel('Eventos', fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

legenda = plt.legend(fontsize=25)
for texto_legenda in legenda.get_texts():
    texto_legenda.set_fontsize(25)



plt.text(0.88, 1.04, "(Z → ℓ⁺ℓ⁻)", transform=plt.gca().transAxes, fontsize=25, ha='left')
plt.show()