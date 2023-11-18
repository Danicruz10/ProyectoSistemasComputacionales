import gym
from gym import spaces
import numpy as np
from .ModeloDelConocimiento.BKT import BKT
import joblib

emotions = ['Neutral', 'Happy', 'Sad', 'Fear', 'Disgust', 'Surprise', 'Anger', 'Contempt']  # Se definen las emociones 
skills = ['1', '2', '3', '4', '5']  # Se definen las habilidades (categorías de preguntas)
filename = 'trained_model'
loaded_model = joblib.load(filename)


class StudentEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.action_space = spaces.Discrete(len(skills))
        self.observation_space = spaces.Discrete(len(emotions))
        self.bkt_models = {skill: BKT(str(skill)) for skill in skills} 
        self.reset()

    def reset(self, seed=None, options=None):
        # Reiniciar el estado emocional al azar
        self.action = np.random.randint(len(skills))
        self.reward = 0.0
        self.state = np.random.randint(len(emotions))
        return self.state, {}
       
    def step(self, action):

        skill = skills[action]
        bkt_model = self.bkt_models[skill]
        probability = bkt_model.probabilty() 
        # Calcular la recompensa basada en la probabilidad de respuesta correcta
        correctAnswer = np.random.choice([0, 1], p=[1 - probability , probability])
        self.reward = 1 if correctAnswer else 0 #operador ternario, si respuesta correcta es true devuelve 1 de lo contrario 0

        # Cambiar al siguiente estado
        self.state = loaded_model.predict(np.array([[action, correctAnswer, self.state]]))[0]
        

        return self.state, self.reward, False, False, {}
    
    def render(self):
        print(
            "Action {}, reward {}, state {}".format(
                self.action, self.reward, self.state
            )
        )
            
env = StudentEnv()
        
        





        
