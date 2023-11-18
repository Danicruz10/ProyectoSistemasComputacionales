from gym.envs.registration import register
import gym
from agent import QLearning


register(
    id="Student-v1",
    entry_point="ReinforcementLearning-master.environment:StudentEnv"
)

# Crear una instancia del entorno
env = gym.make('Student-v1') 
agent = QLearning(env.observation_space.n, env.action_space.n, alpha = 0.1, gamma = 0.9, epsilon = 0.1)

# Comenzar un nuevo episodio
initState, _ = env.reset()

for _ in range(100):
   # Bucle para iterar a través de los pasos de tiempo del episodio
    done = False

    # Tomar una acción (elegir una habilidad)
    action = agent.get_action(initState, 'epsilon-greedy')
    # Ejecutar la acción y obtener el nuevo estado y recompensa
    next_state, reward, terminated, _, _ = env.step(action)
    # Actualizacion de tabla Qtable
    agent.update(initState, action, next_state, reward, terminated)
    # Guardando el estado para la siguiente iteracion
    initState = int(next_state)

    agent.render()
    agent.render('step')

    # Fin del episodio
