# alinea_agent.py

class ALINEA_Agent:
    def __init__(self, K, q_target, initial_rate=1.0):
        self.K = K
        self.q_target = q_target
        self.r = initial_rate

    def update_rate(self, q_current):
        self.r += self.K * (self.q_target - q_current)
        return self.r
