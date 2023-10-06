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
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')

pasta_csv = r"C:\Users\venut\Desktop\CSV"
# Lista para armazenar os caminhos completos de todos os arquivos CSV na pasta
arquivos_csv = []
#envia os arquivos da pasta para a lista criada
for arquivo in os.listdir(pasta_csv):
         arquivos_csv.append(os.path.join(pasta_csv, arquivo))
#junta os arquivos em 1 só
df = pd.concat((pd.read_csv(file_path) for file_path in arquivos_csv), axis=0)





def mass_zz(row):
    v1 = vector.obj(px=row['px1'], py=row['py1'], pz=row['pz1'], E=row['E1'])
    v2 = vector.obj(px=row['px2'], py=row['py2'], pz=row['pz2'], E=row['E2'])
    v3 = vector.obj(px=row['px3'], py=row['py3'], pz=row['pz3'], E=row['E3'])
    v4 = vector.obj(px=row['px4'], py=row['py4'], pz=row['pz4'], E=row['E4'])
    a1 = v1.pt
    b1 = v2.pt
    c1 = v3.pt
    d1 = v4.pt
    dpt1= abs(a1) - abs(b1)
    dpt2= abs(a1) - abs(c1)
    dpt3= abs(a1) - abs(d1)
    ideal= 91.19  
    a2 = (v1+v2).mass
    b2 = (v1+v3).mass
    c2 = (v1+v4).mass
    dm1= ideal - a2
    dm2= ideal - b2
    dm3= ideal - c2


    if  abs(row['PID1']) == abs(row['PID2']) and abs(row['PID1']) != abs(row['PID3']) and row['Q1']!=row['Q2']:
        return (v1+v2).mass , (v3+v4).mass


    if row['Q1'] != row['Q2'] and row['Q1'] != row['Q3'] and abs(row['PID1']) == abs(row['PID2']) and abs(row['PID1']) == abs(row['PID3']) and b2<103:
        if dpt1<dpt2 and dm1<dm2 and 10 < a2 < 103.5:
           return (v1+v2).mass , (v3+v4).mass
        else:
             return (v1+v3).mass , (v2+v4).mass    
    if row['Q1'] != row['Q2'] and row['Q1'] != row['Q4'] and abs(row['PID1']) == abs(row['PID2']) and abs(row['PID1']) == abs(row['PID4']) and c2 < 103 :
        if dpt1<dpt3 and dm1<dm3 and 10 < a2 < 103.5:
           return (v1+v2).mass , (v3+v4).mass
        else:
             return (v1+v4).mass , (v2+v3).mass
    

    elif row['Q1'] == row['Q2'] and abs(row['PID1']) == abs(row['PID3']) and row['Q1']!=row['Q3'] and abs(row['PID1']) == abs(row['PID4']) and c2<103:
        if dpt2<dpt3 and dm2<dm3 and 10 < b2 < 103.5:
           return (v1+v3).mass , (v2+v4).mass
        else:
             return (v1+v4).mass , (v2+v3).mass
        
    else:
        return 0 , 0 

        
df['massa1'],df['massa2'] = zip(*df.apply(mass_zz,axis=1))



x1data1=df['massa1'].dropna().values
x1data2=df['massa2'].dropna().values
x1data= np.hstack((x1data1, x1data2))


x2data1=df['mZ1']
x2data2=df['mZ2']
x2data=np.hstack((x2data1, x2data2))


a=44
plt.hist(x1data, bins=a,alpha=0.4, label='elétron', color='darkgreen')
plt.hist(x2data, bins=a, alpha=0.7,label='pósitron', color='darkblue')
print(x1data)
plt.show()







