import numpy as np

class CumulativeFatigue():
    # 3CC-r model, adapted from https://dl.acm.org/doi/pdf/10.1145/3313831.3376701 for muscles 
    # based on the implementation from Aleksi Ikkala and Florian Fischer https://github.com/aikkala/user-in-the-box/blob/main/uitb/bm_models/effort_models.py
    def __init__(self, mj_model):
        self._r = 15 # Recovery time multiplier i.e. how many times more than during rest intervals https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6092960/
        self._F = 0.0146 # Fatigue coefficients
        self._R = 0.0022 # Recovery coefficients
        self.na = mj_model.na
        self._dt = mj_model.opt.timestep
        self._LD = np.array([1/mj_model.actuator_dynprm[i][0] for i in range(self.na)])
        self._LR = np.array([1/mj_model.actuator_dynprm[i][1] for i in range(self.na)])
        self._MA = np.zeros((self.na,))  # Muscle Active
        self._MR = np.ones((self.na,))   # Muscle Resting
        self._MF = np.zeros((self.na,))  # Muscle Fatigue
        self.TL  = np.zeros((self.na,))  # Target Load

    def set_FatigueCoefficient(self, F):
        # Set Fatigue coefficients
        self._F = F
    
    def set_RecoveryCoefficient(self, R):
        # Set Recovery coefficients
        self._R = R
    
    def set_RecoveryMultiplier(self, r):
        # Recovery time multiplier
        self._r = r
        
    def compute_act(self, act):
        # Get target load (actual activation, which might be reached only with some "effort", 
        # depending on how many muscles can be activated (fast enough) and how many are in fatigue state)
        self.TL = act 

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

        # Calculate MA, MR
        self._MA += (C - self._F*self._MA)*self._dt
        self._MR += (-C + rR*self._MR)*self._dt
        self._MF += (self._F*self._MA - rR*self._MF)*self._dt

        # Not sure if these are needed
        self._MA = np.clip(self._MA, 0, 1)
        self._MR = np.clip(self._MR, 0, 1)
        self._MF = np.clip(self._MF, 0, 1)

        return self._MA, self._MR, self._MF

    def get_effort(self):
        # Calculate effort
        return np.linalg.norm(self._MA - self.TL)

    def reset(self):
        self._MA = np.zeros((self.na,))  # Muscle Active
        self._MR = np.ones((self.na,))   # Muscle Resting
        self._MF = np.zeros((self.na,))  # Muscle Fatigue

