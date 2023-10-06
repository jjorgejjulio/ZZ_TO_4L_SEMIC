#Bibliotecas utilizadas
import pandas as pd
import vector
import numpy as np
import matplotlib.pyplot as plt  
import math
import csv
from scipy.optimize import curve_fit
import os
from scipy.interpolate import make_interp_spline

#Arquivo csv contendo as informações de cada partícula (4 leptons)
pasta_csv = r"C:\Users\venut\Desktop\CSV"
# Lista para armazenar os caminhos completos de todos os arquivos CSV na pasta
arquivos_csv = []
#envia os arquivos da pasta para a lista criada
for arquivo in os.listdir(pasta_csv):
         arquivos_csv.append(os.path.join(pasta_csv, arquivo))
#junta os arquivos em 1 só
df = pd.concat((pd.read_csv(file_path) for file_path in arquivos_csv), axis=0)


c=11

def compute_pt1(row):
    v1 = vector.obj(px=row['px1'], py=row['py1'], pz=row['pz1'], E=row['E1'])
    if abs(row['PID1']) == c :
      return v1.pt

def compute_pt2(row):
    v2 = vector.obj(px=row['px2'], py=row['py2'], pz=row['pz2'], E=row['E2'])
    if abs(row['PID2'])==c:
     return v2.pt
 
def compute_pt3(row):
    v3 = vector.obj(px=row['px3'], py=row['py3'], pz=row['pz3'], E=row['E3'])
    if abs(row['PID3'])==c:
     return v3.pt

def compute_pt4(row):
    v4 = vector.obj(px=row['px4'], py=row['py4'], pz=row['pz4'], E=row['E4'])
    if abs(row['PID4'])==c:
     return v4.pt

df['pt1'] = df.apply(compute_pt1, axis=1)
df['pt2'] = df.apply(compute_pt2, axis=1)
df['pt3'] = df.apply(compute_pt3, axis=1)
df['pt4'] = df.apply(compute_pt4, axis=1)

x1data=np.array(df['pt1'])
x2data=np.array(df['pt2'])
x3data=np.array(df['pt3'])
x4data=np.array(df['pt4'])



x1data = x1data[~np.isnan(x1data)]
x2data = x2data[~np.isnan(x2data)]
x3data = x3data[~np.isnan(x3data)]
x4data = x4data[~np.isnan(x4data)]


#bins
a=13
b=5


#Pt das partícula1 
hist1,bins1=np.histogram(x1data,bins=a)
x1= (bins1[:-1] + bins1[1:]) / 2 
y1=hist1
#Pt das partícula2
hist2,bins2=np.histogram(x2data,bins=a)
x2= (bins2[:-1] + bins2[1:]) / 2 
y2=hist2
#Pt das partícula3
hist3,bins3=np.histogram(x3data,bins=a)
x3= (bins3[:-1] + bins3[1:]) / 2 
y3=hist3
#Pt das partícula4
hist4,bins4=np.histogram(x4data,bins=a)
x4= (bins4[:-1] + bins4[1:]) / 2 
y4=hist4



plt.figure(figsize=(6,6))
plt.rcParams['font.size'] = 25


X1_Y1_Spline = make_interp_spline(x1, y1)
X1_ = np.linspace(x1.min(), x1.max(), 65)
Y1_ = X1_Y1_Spline(X1_)

X2_Y2_Spline = make_interp_spline(x2, y2)
X2_ = np.linspace(x2.min(), x2.max(), 65)
Y2_ = X2_Y2_Spline(X2_)

X3_Y3_Spline = make_interp_spline(x3, y3)
X3_ = np.linspace(x3.min(), x3.max(), 65)
Y3_ = X3_Y3_Spline(X3_)

X4_Y4_Spline = make_interp_spline(x4, y4)
X4_ = np.linspace(x4.min(), x4.max(), 65)
Y4_ = X4_Y4_Spline(X4_)

#plots linhas
plt.plot(X1_, Y1_,  label='partícula 1',color='black')
plt.plot(X2_, Y2_,  label='partícula 2',color='blue')
plt.plot(X3_, Y3_,  label='partícula 3',color='red')
plt.plot(X4_, Y4_,  label='partícula 4',color='purple')

#plots steps
#plt.step(x1, hist1,  label='partícula 1',color='black')
#plt.step(x2, hist2,  label='partícula 2',color='blue')
#plt.step(x3, hist3,  label='partícula 3',color='red')
#plt.step(x4, hist3,  label='partícula 4',color='purple')


#plots histogramas
#plt.hist(x1data,  bins=a,width=b, alpha=0.4, label='Partícula 1', color='darkgreen')
#plt.hist(x2data, bins=a ,width=b,alpha=0.7,label='Partícula 2', color='darkblue')
#plt.hist(x3data,bins=a , width=b,alpha=0.5, label='Partícula 3', color='darkred')
#plt.hist(x4data, bins=a ,width=b, alpha=0.6,label='Partícula 4', color='purple')



#legendas e títulos
plt.title('$ \sqrt{s} = 7$ TeV, L = 2.3 $fb^{-1}$; $\sqrt{s} = 8$ TeV, L = 11.6 $fb^{-1}$ \n', fontsize = 25)
plt.xlabel('Momento transverso (Pt) GeV', fontsize = 25)
plt.ylabel('Eventos', fontsize = 25)
plt.text(0.88, 1.04, "(e⁺e⁻)", transform=plt.gca().transAxes, fontsize=25, ha='left')
plt.legend()
plt.xlim(20,150)

plt.show()



