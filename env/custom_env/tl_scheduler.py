class TlScheduler:
    def __init__(self, tp_min, tl_ids):
        # Initialize the scheduler with the minimum time period and traffic light IDs
        self.idx = 0  # Index to track the current time step
        self.size = tp_min + 1  # Size of the buffer based on the time period
        # Initialize the buffer as a list of empty lists
        self.buffer = [[] for _ in range(self.size)]
        # Push initial events with zero time event for each traffic light ID
        [self.push(0, (tl_id, None)) for tl_id in tl_ids]

    def push(self, t_evt, tl_evt):
        # Add a traffic light event to the buffer at the specified time event
        self.buffer[(self.idx + t_evt) % self.size].append(tl_evt)

    def pop(self):
        # Retrieve and remove the next traffic light event from the buffer
        try:
            tl_evt = self.buffer[self.idx].pop(0)
        except IndexError:
            # If the buffer is empty, move to the next time step
            tl_evt = None
            self.idx = (self.idx + 1) % self.size
        return tl_evt
