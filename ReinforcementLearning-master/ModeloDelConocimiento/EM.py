# Paso 1: Importar los módulos necesarios
from pyBKT.models import Model
import numpy as np
import pandas as pd


#Paso 2: Crear una instancia del modelo pyBKT
model = Model()

# Paso 3: Ajustar el modelo con tus datos
model.fit(data_path = '../../NewDatasetC.csv')

# Lista de habilidades (asumo que las habilidades son identificadas por números, ajusta según sea necesario)
skill_id = ['1', '2', '3', '4', '5']

all_params = {}#diccionario de parametros

for skill in skill_id:
    skills_Params = model.params().loc[(skill)]
    all_params[skill] = { 
                    'P(L)': skills_Params.loc[('prior','default')]['value'], # prior
                    'P(T)': skills_Params.loc[('learns','default')]['value'], # learns
                    'P(G)': skills_Params.loc[('guesses','default')]['value'],# Guesses
                    'P(S)': skills_Params.loc[('slips','default')]['value'],#slips
                }
    


Parmetros = all_params

