class TWAPStrategy:
    def generate_schedule(self, total_shares, time_periods):
        return [total_shares / time_periods] * time_periods

class FrontLoadedStrategy:
    def generate_schedule(self, total_shares, time_periods):
        weights = [2**(-i/5) for i in range(time_periods)]
        total_weight = sum(weights)
        return [w * total_shares / total_weight for w in weights]

class BackLoadedStrategy:
    def generate_schedule(self, total_shares, time_periods):
        weights = [2**(i/5) for i in range(time_periods)]
        total_weight = sum(weights)
        return [w * total_shares / total_weight for w in weights]