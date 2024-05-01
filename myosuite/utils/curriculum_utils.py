
class curriculum():
    """
    Set up an curriculum factoring the current progress of agents
    """
    def __init__(self,
                threshold = 90.0,   # value above which curriculum is active
                rate = 1.0/100.0,   # rate of progress for curriculum
                start = 0.0,        # starting value of curriculum
                end = 1.0,          # ending value of curriculum
                filter_coef = 0.95, # filter for updating the progress
                ):

        self._threshold = threshold
        self._rate = rate
        self._start = start
        self._end = end
        self._filter_coef = filter_coef

        self._value = 0.0           # curriculum's current value
        self._progress = 0.0        # curriculum's measure of overall progress

        assert self._rate>0, "rate should always be positive"

    # update the curriculum based on current progress made by the agent
    def update(self, current_success):
        # update the progress measure
        self._progress = self._progress*self._filter_coef + current_success*(1.-self._filter_coef)

        # if sufficient progress, bump curriculum
        if self._value <= 1.0: # if not saturated
            if(current_success>=self._threshold): # if maintaining quality
                if(self._progress>=self._threshold): # if progress is satisfactory
                    self._value += self._rate

    # get the current curriculum status
    def status(self):
        return self._start + self._value*(self._end - self._start)