# -*- coding: utf-8 -*-
import pandas as pd
from BKT import BKT
from EM import Parmetros


# Leer el conjunto de datos
#Extraer valores únicos de la columna correct
df = pd.read_csv('../../DatasetC.csv')



unique_skill = df['skill_id'].unique()
for skill in unique_skill:
    student_model_agent = BKT(str(skill))
    corrects = df['correct'] #150 skill por estudiante
    i=1
    for correct in corrects:
        print(f"iteracion: {i}: de la skill:{skill} su respuesta a esa skill fue: {correct}, probabilidad de contestar correctamente la proxima vez:{student_model_agent.Predict(correct)}")
        i+=1
    
    
