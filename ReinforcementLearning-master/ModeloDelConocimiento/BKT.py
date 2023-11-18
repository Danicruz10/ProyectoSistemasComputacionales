import pandas as pd
import numpy as np
from .EM import Parmetros



class BKT:
    def __init__(self, skill): 
        self.p_L_obs = 0 
        self.p_L_0 = Parmetros[skill]['P(L)'] #P(L) se actualiza en el tiempo segun la formula
        self.p_L = self.p_L_0 # se inicializa pl igual a pl0
        self.p_T = Parmetros[skill]['P(T)']
        self.p_G = Parmetros[skill]['P(G)']
        self.p_S = Parmetros[skill]['P(S)']
        self.p_Ct = self.p_L*(1-(self.p_S)) + (1-(self.p_L))*self.p_G #se inicializa con los valores iniciales de los parametros de cada skill
        

    def Predict(self,correctAnswer): 
        if correctAnswer == 1:  # si es correcto
            self.p_L_obs = ((self.p_L* (1 - self.p_S)) / ((self.p_L * (1 - self.p_S)) + ((1 - self.p_L) * self.p_G)))
        else:  # si es incrrecto
            self.p_L_obs = ((self.p_L * self.p_S) / ((self.p_L * self.p_S) + ((1 - self.p_L) * (1 - self.p_G))))
        self.p_L = (self.p_L_obs + ((1 - self.p_L_obs) * self.p_T))
        
        self.p_Ct = self.p_L*(1-(self.p_S)) + (1-(self.p_L))*self.p_G
        return self.p_Ct
    
    def probabilty(self):
        return self.p_Ct