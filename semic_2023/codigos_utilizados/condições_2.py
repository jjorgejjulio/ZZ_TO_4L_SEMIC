#Bibliotecas utilizadas
import pandas as pd
import vector
import numpy as np
import matplotlib.pyplot as plt  
import csv
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
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


#Definição da função que faz as condições de combinação e encontra a massa invariante de Z1
def compute_mass1(row):
    v1 = vector.obj(px=row['px1'], py=row['py1'], pz=row['pz1'], E=row['E1'])
    v2 = vector.obj(px=row['px2'], py=row['py2'], pz=row['pz2'], E=row['E2'])
    v3 = vector.obj(px=row['px3'], py=row['py3'], pz=row['pz3'], E=row['E3'])
    v4 = vector.obj(px=row['px4'], py=row['py4'], pz=row['pz4'], E=row['E4'])
    ideal= row['mZ1'] 
    a = abs(ideal -(v1+v2).mass)
    b = abs(ideal -(v1+v3).mass)
    c = abs(ideal -(v1+v4).mass)
    lista1= [a,b]
    lista2= [a,c]
    lista3= [b,c]
    menor1=min(lista1)
    menor2=min(lista2)
    menor3=min(lista3)

    if abs(row['PID1']) == abs(row['PID3']) and abs(row['PID1']) == abs(row['PID4']) and row['Q1'] == row['Q2'] :
        if b==menor3 and (v1+v3).mass < 110 :
           return (v1+v3).mass
        else:
             return (v1+v4).mass


    if abs(row['PID1']) == abs(row['PID2']) and abs(row['PID1']) == abs(row['PID3']) and row['Q1'] != row['Q2'] and row['Q1'] == row['Q4'] :
        if a==menor1 and (v1+v2).mass < 110:
           return (v1+v2).mass
        else:
             return (v1+v3).mass
        

    if abs(row['PID1']) == abs(row['PID2']) and abs(row['PID1']) == abs(row['PID4']) and row['Q1'] != row['Q2'] and row['Q1'] == row['Q3'] :
        if a==menor2 and (v1+v2).mass < 110:
           return (v1+v2).mass
        else:
             return (v1+v4).mass


    if row['Q1'] != row['Q2'] and abs(row['PID1']) == abs(row['PID2']) and abs(row['PID1']) != abs(row['PID3']):
         return (v1+v2).mass
    
df['massa1'] = df.apply(compute_mass1, axis=1)



#Definição da função que faz as condições de combinação e encontra a massa invariante de Z2
def compute_mass2(row):
    v1 = vector.obj(px=row['px1'], py=row['py1'], pz=row['pz1'], E=row['E1'])
    v2 = vector.obj(px=row['px2'], py=row['py2'], pz=row['pz2'], E=row['E2'])
    v3 = vector.obj(px=row['px3'], py=row['py3'], pz=row['pz3'], E=row['E3'])
    v4 = vector.obj(px=row['px4'], py=row['py4'], pz=row['pz4'], E=row['E4'])
    ideal=  row['mZ2'] 
    a = abs(ideal -(v3+v4).mass)
    b = abs(ideal -(v2+v3).mass)
    c = abs(ideal -(v2+v4).mass)
    lista1= [a,b]
    lista2= [a,c]
    lista3= [b,c]
    menor1=min(lista1)
    menor2=min(lista2)
    menor3=min(lista3)

    if abs(row['PID3']) == abs(row['PID4']) and abs(row['PID3']) == abs(row['PID2']) and row['Q3'] == row['Q4'] :
        if b==menor3 and (v3+v2).mass < 110 :
           return (v3+v2).mass
        else:
             return (v4+v2).mass


    if abs(row['PID3']) == abs(row['PID4']) and abs(row['PID4']) == abs(row['PID2']) and row['Q3'] != row['Q4'] and row['Q3'] == row['Q2'] :
        if a==menor2 and (v3+v4).mass < 110:
           return (v3+v4).mass
        else:
             return (v4+v2).mass
        

    if abs(row['PID3']) == abs(row['PID4']) and abs(row['PID3']) == abs(row['PID2']) and row['Q3'] != row['Q4'] and row['Q4'] == row['Q2'] :
        if a==menor1 and (v3+v4).mass < 110:
           return (v3+v4).mass
        else:
             return (v3+v2).mass


    if row['Q3'] != row['Q4'] and abs(row['PID3']) == abs(row['PID4']) and abs(row['PID3']) != abs(row['PID2']):
         return (v3+v4).mass
df['massa2'] = df.apply(compute_mass2, axis=1)



# Crio um array com os valores encontrados com as funções e em x3 uno os array
x1data=np.array(df['massa1'])
x2data=np.array(df['massa2'])
x3data= np.hstack((x1data, x2data))






# Coleto dos aquivos csv os resultados encontrados para a massa invariante do Z1 e Z2 e em x6 uno os resultados em um único array
x4data=np.array(df['mZ1'])
x5data=np.array(df['mZ2'])
x6data=np.hstack((x4data, x5data))

# Faço um plot com os resultados calculados e com o resultado esperado do arquivo CSV e comparo
plt.figure(figsize=(6,6))
plt.hist(x3data,label='Z calculado',alpha=0.5,bins=15, color='blue')
plt.hist(x6data,label='Z referência',alpha=0.3,bins=15, color='red')
plt.xlim(0,180)
plt.legend()
plt.show()

print(df[['massa1']])
