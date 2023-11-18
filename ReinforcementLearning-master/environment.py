import gym
from gym import spaces
import numpy as np

emotions = ['sorpresa', 'alegria', 'asco', 'tristeza', 'ira', 'miedo']  # Se definen las emociones 
skills = ['presente_simple', ' proposiciones', 'pronombres_sujetos', 'adverbios_de_frecuencia', 'Artículos']  # Se definen las habilidades (categorías de preguntas)


class StudentEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.action_space = spaces.Discrete(len(skills))
        self.observation_space = spaces.Discrete(len(emotions))
        self.P = {
            0:{ # Probabilidades de contestar correctamente cuando está con sorpresa
                0: 0.3, # presente_simple
                1: 0.7,# proposicione
                2: 0.5,#pronombres_sujetos
                3: 0.4,#adverbios_de_frecuencia
                4: 0.6,#Artículos
            },
            1:{ # Probabilidades de contestar correctamente cuando está con alegria
                0: 0.8,# presente_simple
                1: 0.9,# proposicione
                2: 0.7,#pronombres_sujetos
                3: 0.6,#adverbios_de_frecuencia
                4: 0.9,#Artículos
            },
            2:{ # Probabilidades de contestar correctamente cuando está con asco
                0: 0.3, # presente_simple
                1: 0.4,# proposicione
                2: 0.2,#pronombres_sujetos
                3: 0.4,#adverbios_de_frecuencia
                4: 0.3,#Artículos
            },
            3:{ # Probabilidades de contestar correctamente cuando está con tristeza
                0: 0.2, # presente_simple
                1: 0.3,# proposicione
                2: 0.1,#pronombres_sujetos
                3: 0.3,#adverbios_de_frecuencia
                4: 0.4,#Artículos
            },
            4:{ # Probabilidades de contestar correctamente cuando está con ira
                0: 0.7, # presente_simple
                1: 0.4,# proposicione
                2: 0.7,#pronombres_sujetos
                3: 0.6,#adverbios_de_frecuencia
                4: 0.5,#Artículos
            },
            5:{ # Probabilidades de contestar correctamente cuando está con miedo
                0: 0.6, # presente_simple
                1: 0.5,# proposicione
                2: 0.1,#pronombres_sujetos
                3: 0.8,#adverbios_de_frecuencia
                4: 0.6,#Artículos
            },
        }
        self.reset()

    def reset(self, seed=None, options=None):
        # Reiniciar el estado emocional al azar
        self.action = np.random.randint(len(skills))
        self.reward = 0.0
        self.state = np.random.randint(len(emotions))
        return self.state, {}
       
    def step(self, action):
        # Calcular la recompensa basada en la probabilidad de respuesta correcta
        correctProbability = self.P[self.state][action]
        print(correctProbability)
        correctAnswer = np.random.choice([True, False], p=[correctProbability, 1 - correctProbability])
        self.reward = 1 if correctAnswer else 0 #operador ternario, si respuesta correcta es true devuelve 1 de lo contrario 0

        # Cambiar el estado emocional al azar
        self.state = np.random.randint(len(emotions))

        return self.state, self.reward, False, False, {}
    
    def render(self):
        print(
            "Action {}, reward {}, state {}".format(
                self.action, self.reward, self.state
            )
        )
            
env = StudentEnv()
        
        





        
