from .tl_scheduler import TlScheduler
from .sumo_env import SumoEnv
import traci

class RLController(SumoEnv):
    def __init__(self, *args, **kwargs):
        super(RLController, self).__init__(*args, **kwargs)

        self.tg = 10  # Base green time for ramp meters
        self.tr = 2   # Red time for ramp meters

        self.dtse_shape = self.get_dtse_shape()
        self.sum_delay_sq_min = 0

        self.scheduler, self.next_tl_id = None, None

        self.ramp_meter_ids = ['ramp12', 'ramp14', 'ramp16']  # Example ramp meter IDs
        self.edge_after_ramp = 'E6'  # Example edge ID just after the ramp
        self.action_space_n = 3  # Example: 0 = low rate, 1 = medium rate, 2 = high rate
        self.observation_space_n = self.dtse_shape  # Shape of the state representation

        # Define thresholds
        self.density_threshold = 0.8
        self.flow_threshold = 0.8
        self.max_queue_length = 100

        # Da dictionary mapping tl_id to ramp lane IDs
        self.ramp_lane_mapping = {
            'ramp12': ['ramp12_0'],  # lane IDs for ramp12
            'ramp14': ['ramp14_0'],  #  lane IDs for ramp14
            'ramp16': ['ramp16_0']   #  lane IDs for ramp16
        }


    def reset(self):
        self.simulation_reset()

        self.scheduler = TlScheduler(self.tg + self.tr, self.ramp_meter_ids)
        self.next_tl_id = self.scheduler.pop()[0]

        for _ in range(self.tg):
            self.simulation_step()

    def step(self, action):
        tl_id = self.next_tl_id
        green_time = self.tg * (action + 1)  # Different green times based on the action

        self.set_phase(tl_id, 0)  # Assuming 0 is green phase
        self.set_phase_duration(tl_id, green_time)
        self.scheduler.push(green_time + self.tr, (tl_id, None))

        while True:
            tl_evt = self.scheduler.pop()
            if tl_evt is None:
                self.simulation_step()
            else:
                tl_id, _ = tl_evt
                self.next_tl_id = tl_id
                return
            
    def obs(self):
        tl_id = self.next_tl_id

        density = self.get_density_after_ramp(self.edge_after_ramp)
        flow = self.get_flow_after_ramp(self.edge_after_ramp)
        queue_length = self.get_ramp_queue_length(tl_id)
        speed = self.get_average_speed(self.edge_after_ramp)

        obs = [density, flow, queue_length, speed]
        return obs

    def rew(self):
        tl_id = self.next_tl_id
        sum_delay_sq = self.get_sum_delay_sq(tl_id)
        self.sum_delay_sq_min = min([self.sum_delay_sq_min, -sum_delay_sq])
        rew = 0 if self.sum_delay_sq_min == 0 else 1 + sum_delay_sq / self.sum_delay_sq_min

        # Penalization based on density and flow thresholds
        density_after_ramp = self.get_density_after_ramp(self.edge_after_ramp)
        flow_after_ramp = self.get_flow_after_ramp(self.edge_after_ramp)
        queue_length = self.get_ramp_queue_length(tl_id)

        if density_after_ramp > self.density_threshold:
            rew -= (density_after_ramp - self.density_threshold)
        
        if flow_after_ramp > self.flow_threshold:
            rew -= (flow_after_ramp - self.flow_threshold)
        
        if queue_length > self.max_queue_length:
            rew -= (queue_length - self.max_queue_length)
        
        return SumoEnv.clip(0, 1, rew)

    def done(self):
        return self.is_simulation_end() or self.get_current_time() >= self.args["steps"]

# Connected vehicles

    def get_veh_delay_sq(self, veh_id):
        return 1 - pow((self.get_veh_speed(veh_id) / self.args["v_max_speed"]), 2)


    def get_sum_delay_sq(self, tl_id):
        sum_delay = 0
        for veh_id in self.yield_tl_vehs(tl_id):
            sum_delay += self.get_veh_delay_sq(veh_id)
        return sum_delay

    def get_n_cells(self):
        return self.args["con_range"] // self.args["cell_length"]

    def get_dtse_shape(self):
        return (3, len(self.get_tl_incoming_lanes(self.ramp_meter_ids[0])), self.get_n_cells())

    def get_dtse(self, tl_id):
        if len(self.dtse_shape) < 3:
            raise ValueError("dtse_shape must have at least three dimensions")
    
        dtse = [[[0. for _ in range(self.dtse_shape[2])] for _ in range(self.dtse_shape[1])] for _ in range(self.dtse_shape[0])]
        for l in range(len(dtse[2])):
            dtse[2][l] = [1. for _ in range(self.dtse_shape[2])]
        return dtse

    def get_density_after_ramp(self, edge_id):
        lane_ids = traci.edge.getLaneIDs(edge_id)
        total_vehicles = 0
        total_length = 0
        for lane_id in lane_ids:
            total_vehicles += traci.lane.getLastStepVehicleNumber(lane_id)
            total_length += traci.lane.getLength(lane_id)
        density = total_vehicles / total_length if total_length > 0 else 0
        return density
    
    def get_flow_after_ramp(self, edge_id):
        lane_ids = traci.edge.getLaneIDs(edge_id)
        total_flow = 0
        for lane_id in lane_ids:
            total_flow += traci.lane.getLastStepVehicleNumber(lane_id)  # Number of vehicles in the last step
        return total_flow

    def get_ramp_queue_length(self, tl_id):
        ramp_lane_ids = self.get_ramp_lane_ids(tl_id)
        queue_length = 0
        for lane_id in ramp_lane_ids:
            veh_ids = traci.lane.getLastStepVehicleIDs(lane_id)
            for veh_id in veh_ids:
                if traci.vehicle.getSpeed(veh_id) < 0.1:  # Assuming vehicles with speed < 0.1 are queuing
                    queue_length += 1
        return queue_length

    def get_ramp_lane_ids(self, tl_id):
        """
        This method returns the list of lane IDs associated with the given ramp traffic light ID (tl_id).
        """
        if tl_id in self.ramp_lane_mapping:
            return self.ramp_lane_mapping[tl_id]
        else:
            raise ValueError(f"Invalid traffic light ID: {tl_id}")


    def get_average_speed(self, edge_id):
        lane_ids = traci.edge.getLaneIDs(edge_id)
        total_speed = 0
        total_vehicles = 0
        for lane_id in lane_ids:
            veh_ids = traci.lane.getLastStepVehicleIDs(lane_id)
            for veh_id in veh_ids:
                total_speed += traci.vehicle.getSpeed(veh_id)
                total_vehicles += 1
        avg_speed = total_speed / total_vehicles if total_vehicles > 0 else 0
        return avg_speed
