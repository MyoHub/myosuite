import matplotlib.pyplot as plt
import numpy as np

from myosuite.utils import gym

env = gym.make("myoElbowPose1D6MRandom-v0", normalize_act=False)
envFatigue = gym.make("myoFatiElbowPose1D6MRandom-v0", normalize_act=False)

env.reset()
envFatigue.reset()
data_store = []
data_store_f = []
for i in range(20):  # 10 episodes
    a = np.zeros(
        env.mj_model.na,
    )
    if i % 3:
        a[3:] = 1
    else:
        a[:] = 0

    for _ in range(500):  # 100 samples for each episode
        next_o, r, done, *_, ifo = env.step(a)  # take an action
        next_f_o, r_f, done_F, *_, ifo_f = envFatigue.step(a)  # take an action

        data_store.append(
            {
                "action": a.copy(),
                "jpos": env.mj_data.qpos.copy(),
                "mlen": env.mj_data.actuator_length.copy(),
                "act": env.mj_data.act.copy(),
            }
        )
        data_store_f.append(
            {
                "action": a.copy(),
                "jpos": envFatigue.mj_data.qpos.copy(),
                "mlen": envFatigue.mj_data.actuator_length.copy(),
                "act": envFatigue.mj_data.act.copy(),
            }
        )

env.close()
envFatigue.close()

plt.figure()
plt.subplot(221), plt.plot(np.array([d["jpos"] for d in data_store])), plt.title(
    "Normal model"
)
plt.xlabel("time"), plt.ylabel("angle")
plt.subplot(222), plt.plot(np.array([d["jpos"] for d in data_store_f])), plt.title(
    "Fatigued model"
)
plt.xlabel("time"), plt.ylabel("angle")
plt.subplot(212), plt.plot(
    np.array([d["jpos"] for d in data_store])
    - np.array([d["jpos"] for d in data_store_f])
)
plt.xlabel("time"), plt.ylabel("angle difference")
plt.title("Difference between normal vs fatigue")
plt.show()
plt.show()
