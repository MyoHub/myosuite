from myosuite.utils import gym
import numpy as np
import matplotlib.pyplot as plt 

env        = gym.make('myoElbowPose1D6MRandom-v0',normalize_act=False)
envFatigue = gym.make('myoFatiElbowPose1D6MRandom-v0',normalize_act=False)

env.reset()
envFatigue.reset();
data_store = []
data_store_f = []
for i in range(20): # 10 episodes
    a = np.zeros(env.sim.model.na,)
    if i%3:
        a[3:]=1
    else:
        a[:]=0
    
    for _ in range(500): # 100 samples for each episode
        next_o, r, done, *_, ifo = env.step(a) # take an action
        next_f_o, r_f, done_F, *_, ifo_f = envFatigue.step(a) # take an action
                    
        data_store.append({"action":a.copy(), 
                            "jpos":env.sim.data.qpos.copy(), 
                            "mlen":env.sim.data.actuator_length.copy(), 
                            "act":env.sim.data.act.copy()})
        data_store_f.append({"action":a.copy(), 
                            "jpos":envFatigue.sim.data.qpos.copy(), 
                            "mlen":envFatigue.sim.data.actuator_length.copy(), 
                            "act":envFatigue.sim.data.act.copy()})

env.close()
envFatigue.close()

plt.figure()
plt.subplot(221),plt.plot(np.array([d['jpos'] for d in data_store])), plt.title('Normal model')
plt.xlabel('time'),plt.ylabel('angle')
plt.subplot(222),plt.plot(np.array([d['jpos'] for d in data_store_f])), plt.title('Fatigued model')
plt.xlabel('time'),plt.ylabel('angle')
plt.subplot(212),plt.plot(np.array([d['jpos'] for d in data_store])-np.array([d['jpos'] for d in data_store_f]))
plt.xlabel('time'),plt.ylabel('angle difference')
plt.title('Difference between normal vs fatigue')
plt.show()
plt.show()