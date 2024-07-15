class AlineaController:
    def __init__(self, k=0.2, critical_occupancy=0.3):
        self.k = k
        self.critical_occupancy = critical_occupancy
        self.ramp_rates = {}

    def update_merge_rate(self, ramp, current_occupancy, merge_rate0):
        desired_rate = merge_rate0 + self.k * (self.critical_occupancy - current_occupancy)
        self.ramp_rates[ramp] = max(0, desired_rate)

    def get_ramp_rate(self, ramp):
        return self.ramp_rates.get(ramp, 0)
