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
    v1 =vector.obj(px=row['px1'], py=row['py1'], pz=row['pz1'], E=row['E1'])
    v2 =vector.obj(px=row['px2'], py=row['py2'], pz=row['pz2'], E=row['E2'])
    v3 =vector.obj(px=row['px3'], py=row['py3'], pz=row['pz3'], E=row['E3'])
    v4 =vector.obj(px=row['px4'], py=row['py4'], pz=row['pz4'], E=row['E4'])
    return (v1+v2+v3+v4).mass
df['massa1'] = df.apply(compute_mass1, axis=1)
x1data=np.array(df['massa1'])
x2data=np.array(df['M'])

print(x1data.mean())


hist,bins=np.histogram(x1data,bins=100)
x1= (bins[:-1] + bins[1:]) / 2 
y=hist



plt.figure(figsize=(6,6))
plt.scatter(x1, hist,  label='H calculado',color='black',marker='o', s=50)
#plt.hist(x1data, bins=110, label='H calculado', color='blue')
plt.xlim(50, 300)
#plt.ylim(0,25)
plt.suptitle('Higgs')
plt.title('$ \sqrt{s} = 7$ TeV, L = 2.3 $fb^{-1}$; $\sqrt{s} = 8$ TeV, L = 11.6 $fb^{-1}$ \n', fontsize = 12)
plt.xlabel('Massa invariante (GeV)', fontsize = 15)
plt.ylabel('Frequência', fontsize = 15)
plt.legend()
plt.show()





#v1=vector.obj(px=-45.472100, py=-8.610100, pz=-1.240720, E=46.2967)
#v2=vector.obj(px=43.536700, py=14.370800, pz=-0.940301, E=45.85680)
#v3=vector.obj(px=-13.673400	, py=-19.659700, pz=13.016400, E=27.2561)
#v4=vector.obj(px=-0.091305	, py=7.280830, pz=-2.077270, E=7.57191)

#a=(v1+v2+v3+v4).mass
#print(a)


