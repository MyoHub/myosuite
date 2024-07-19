from myosuite.utils import gym
import mujoco
import numpy as np

class CumulativeFatigue():
    # 3CC-r model, adapted from https://dl.acm.org/doi/pdf/10.1145/3313831.3376701 for muscles 
    # based on the implementation from Aleksi Ikkala and Florian Fischer https://github.com/aikkala/user-in-the-box/blob/main/uitb/bm_models/effort_models.py
    def __init__(self, mj_model, frame_skip=1, seed=None):
        self._r = 10 * 15 # Recovery time multiplier i.e. how many times more than during rest intervals https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6092960/ (factor 10 to compensate for 0.1 below)
        self._F = 0.00912  # Fatigue coefficients (default parameter was identified for elbow torque https://pubmed.ncbi.nlm.nih.gov/22579269/)
        self._R = 0.1 * 0.00094  # Recivery coefficients (default parameter was identified for elbow torque https://pubmed.ncbi.nlm.nih.gov/22579269/; factor 0.1 to get an approx. 1% R/F ratio)
        # self.na = mj_model.na
        self._dt = mj_model.opt.timestep * frame_skip # dt might be different from model dt because it might include a frame skip
        muscle_act_ind = mj_model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        self.na = sum(muscle_act_ind)  #WARNING: here, self.na denotes the number of muscle actuators only!
        self._tauact = np.array([mj_model.actuator_dynprm[i][0] for i in range(len(muscle_act_ind)) if muscle_act_ind[i]])
        self._taudeact = np.array([mj_model.actuator_dynprm[i][1] for i in range(len(muscle_act_ind)) if muscle_act_ind[i]])
        self._MA = np.zeros((self.na,))  # Muscle Active
        self._MR = np.ones((self.na,))   # Muscle Resting
        self._MF = np.zeros((self.na,))  # Muscle Fatigue
        self.TL  = np.zeros((self.na,))  # Target Load

        self.seed(seed)  # Create own Random Number Generator (RNG) used when reset is called with fatigue_reset_random=True
        ### NOTE: the seed from CumulativeFatigue is not synchronised with the seed used for the rest of MujocoEnv!

    def set_FatigueCoefficient(self, F):
        # Set Fatigue coefficients
        self._F = F
    
    def set_RecoveryCoefficient(self, R):
        # Set Recovery coefficients
        self._R = R
    
    def set_RecoveryMultiplier(self, r):
        # Set Recovery time multiplier
        self._r = r
        
    def compute_act(self, act):
        # Get target load (actual activation, which might be reached only with some "effort", 
        # depending on how many muscles can be activated (fast enough) and how many are in fatigue state)
        self.TL = act.copy()

        # Calculate effective time constant tau (see https://mujoco.readthedocs.io/en/stable/modeling.html#muscles)
        self._LD = 1/self._tauact*(0.5 + 1.5*self._MA)
        self._LR = (0.5 + 1.5*self._MA)/self._taudeact
        ## TODO: account for smooth transition phase of length tausmooth (tausmooth = mj_model.actuator_dynprm[i][2])?

        # Calculate C(t) -- transfer rate between MR and MA
        C = np.zeros_like(self._MA)
        idxs = (self._MA < self.TL) & (self._MR > (self.TL - self._MA))
        C[idxs] = self._LD[idxs] * (self.TL[idxs] - self._MA[idxs])
        idxs = (self._MA < self.TL) & (self._MR <= (self.TL - self._MA))
        C[idxs] = self._LD[idxs] * self._MR[idxs]
        idxs = self._MA >= self.TL
        C[idxs] = self._LR[idxs] * (self.TL[idxs] - self._MA[idxs])

        # Calculate rR
        rR = np.zeros_like(self._MA)
        idxs = self._MA >= self.TL
        rR[idxs] = self._r*self._R
        idxs = self._MA < self.TL
        rR[idxs] = self._R

        # Clip C(t) if needed, to ensure that MA, MR, and MF remain between 0 and 1
        C = np.clip(C, np.maximum(-self._MA/self._dt + self._F*self._MA, (self._MR - 1)/self._dt + rR*self._MF),
                    np.minimum((1 - self._MA)/self._dt + self._F*self._MA, self._MR/self._dt + rR*self._MF))

        # Update MA, MR, MF
        dMA = (C - self._F*self._MA)*self._dt
        dMR = (-C + rR*self._MF)*self._dt
        dMF = (self._F*self._MA - rR*self._MF)*self._dt
        self._MA += dMA
        self._MR += dMR
        self._MF += dMF

        return self._MA, self._MR, self._MF

    def get_effort(self):
        # Calculate effort
        return np.linalg.norm(self._MA - self.TL)

    def reset(self, fatigue_reset_vec=None, fatigue_reset_random=False):
        if fatigue_reset_random:
            assert fatigue_reset_vec is None, "Cannot use 'fatigue_reset_vec' if fatigue_reset_random=False."
            non_fatigued_muscles = self.np_random.random(size=(self.na,))
            active_percentage = self.np_random.random(size=(self.na,))
            self._MA = non_fatigued_muscles * active_percentage         # Muscle Active
            self._MR = non_fatigued_muscles * (1 - active_percentage)   # Muscle Resting
            self._MF = 1 - non_fatigued_muscles                         # Muscle Fatigue
        else:
            if fatigue_reset_vec is not None:
                assert len(fatigue_reset_vec) == self.na, f"Invalid length of initial/reset fatigue vector (expected {self.na}, but obtained {len(fatigue_reset_vec)})."
                self._MF = fatigue_reset_vec     # Muscle Fatigue
                self._MR = 1 - fatigue_reset_vec # Muscle Resting
                self._MA = np.zeros((self.na,))  # Muscle Active
            else:
                self._MA = np.zeros((self.na,))  # Muscle Active
                self._MR = np.ones((self.na,))   # Muscle Resting
                self._MF = np.zeros((self.na,))  # Muscle Fatigue

    def seed(self, seed=None):
        """
        Set random number seed
        """
        self.input_seed = seed
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    @property
    def MF(self):
        return self._MF
    
    @property
    def MR(self):
        return self._MR
    
    @property
    def MA(self):
        return self._MA
    
    @property
    def F(self):
        return self._F
    
    @property
    def R(self):
        return self._R
    
    @property
    def r(self):
        return self._r


