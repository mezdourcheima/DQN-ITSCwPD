from .tl_scheduler import TlScheduler
from .sumo_env import SumoEnv

import traci
import random

class RLController(SumoEnv):
    def __init__(self, *args, **kwargs):
        super(RLController, self).__init__(*args, **kwargs)

        self.tg = 10  # represent the durations for green, yellow, and red traffic light phases
        self.ty = 3
        self.tr = 2

        #self.edge_after_ramp = 'E6'  # Example edge ID just after the ramp

        self.dtse_shape = self.get_dtse_shape()
        self.sum_delay_sq_min = 0

        self.scheduler, self.next_tl_id = None, None

        self.action_space_n = len(self.tl_logic[self.tl_ids[0]]["act"])  # The number of possible actions (traffic light states) for a traffic light.
        self.observation_space_n = self.dtse_shape  # The dimensions of the state representation

        # Define thresholds
        self.density_threshold = 0.18
        self.flow_threshold = 2.67
        self.max_queue_length = 7 #7 vehicles max in the ramp

        ###############
        # Initialize these attributes
        self.sum_delay_min = float('inf')
        self.sum_waiting_time_min = float('inf')

        # Define the ramp_lane_mapping with the specified ramps
        self.ramp_lane_mapping = {
            'ramp12': ['ramp12_0'],  # lane IDs for ramp12
            'ramp14': ['ramp14_0'],  # lane IDs for ramp14
            'ramp16': ['ramp16_0']   # lane IDs for ramp16
        }

    def reset(self):
        self.simulation_reset()

        self.scheduler = TlScheduler(self.tg + self.ty + self.tr, self.tl_ids)
        self.next_tl_id = self.scheduler.pop()[0]

        for _ in range(self.tg):
            self.simulation_step()

    def step(self, action):
        tl_id = self.next_tl_id

        if self.tl_logic[tl_id]["act"][action] == self.get_ryg_state(tl_id):
            self.scheduler.push(self.tg, (tl_id, None))
            self.set_phase_duration(tl_id, self.tg)
        else:
            for evt in [
                (self.ty, (tl_id, (self.get_next_red_phase_id(tl_id), self.tr))),
                (self.ty + self.tr, (tl_id, (self.get_new_green_phase_id(tl_id, self.tl_logic[tl_id]["act"][action]), self.tg))),
                (self.ty + self.tr + self.tg, (tl_id, None))
            ]:
                self.scheduler.push(*evt)

            self.set_phase(tl_id, self.get_next_yellow_phase_id(tl_id))
            self.set_phase_duration(tl_id, self.ty)

        while True:
            tl_evt = self.scheduler.pop()

            if tl_evt is None:
                self.simulation_step()
            else:
                tl_id, new_p = tl_evt

                if new_p is not None:
                    p, t = new_p
                    self.set_phase(tl_id, p)
                    self.set_phase_duration(tl_id, t)
                else:
                    self.next_tl_id = tl_id
                    return

    def obs(self):
        tl_id = self.next_tl_id
        obs = self.get_dtse(tl_id)
        return obs

    def rew(self):
        total_rew = 0
        tl_id = self.next_tl_id

        # Get the delay and waiting time metrics
        sum_delay, sum_waiting_time = self.get_sum_delay_a_sum_waiting_time(tl_id)

        self.sum_delay_min = min(self.sum_delay_min, -sum_delay)
        self.sum_waiting_time_min = min(self.sum_waiting_time_min, -sum_waiting_time)

        rew_delay = 0 if self.sum_delay_min == 0 else 1 + sum_delay / self.sum_delay_min
        rew_waiting_time = 0 if self.sum_waiting_time_min == 0 else 1 + sum_waiting_time / self.sum_waiting_time_min

        # Weighting factors for delay and waiting time rewards
        w1, w2 = 0.5, 0.5

        rew = w1 * rew_delay + w2 * rew_waiting_time

        for ramp_edge, connected_edges in self.edges_after_ramps.items():
            for edge in connected_edges:
                density = self.get_density(edge)
                flow = self.get_flow(edge)
                queue_length = self.get_ramp_queue_length()

                penalty = 0
                if density > self.density_threshold:
                    penalty += (density - self.density_threshold)
                    print(f"Penalty for density: {density - self.density_threshold} (Density: {density}, Threshold: {self.density_threshold})")
                else:
                    total_rew += 10

                if flow > self.flow_threshold:
                    penalty += (flow - self.flow_threshold)
                    print(f"Penalty for flow: {flow - self.flow_threshold} (Flow: {flow}, Threshold: {self.flow_threshold})")
                else:
                    total_rew += 10

                if queue_length > self.max_queue_length:
                    penalty += (queue_length - self.max_queue_length)
                    print(f"Penalty for queue length: {queue_length - self.max_queue_length} (Queue Length: {queue_length}, Max: {self.max_queue_length})")
                else:
                    total_rew += 10

                total_rew += -penalty
                print(f"Intermediate total reward after edge {edge}: {total_rew}")

        total_rew = max(0, total_rew)  # Ensure the reward is non-negative

        # Incorporate delay and waiting time rewards
        final_rew = SumoEnv.clip(0, 1, rew) + total_rew

        print(f"Final total reward: {final_rew}")
        return final_rew




    def done(self):
        is_done = self.is_simulation_end() or self.get_current_time() >= self.args["steps"]
        # Log episode termination details
        print(f"Episode done: {is_done}, Time: {self.get_current_time()}, Steps: {self.args['steps']}")
        return is_done

    ####################################################################################################################
    ####################################################################################################################

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

    def get_dtse_shape(self):  # shape of the actions space
        return (
            4,  # 4 observations
            len(self.get_tl_incoming_lanes(self.tl_ids[0])),
            20  # fixed number of cells
        )

    def get_dtse(self, tl_id):
        if len(self.dtse_shape) < 3:
            raise ValueError("dtse_shape must have at least three dimensions")

        dtse = [[[0. for _ in range(self.dtse_shape[2])] for _ in range(self.dtse_shape[1])] for _ in range(self.dtse_shape[0])]

        lanes = self.get_tl_incoming_lanes(tl_id)
        for idx, lane in enumerate(lanes):
            edge_id = self.get_lane_edge_id(lane)
            density = self.get_density(edge_id)
            flow = self.get_flow(edge_id)
            queue_length = self.get_ramp_queue_length()
            speed = self.get_average_speed(edge_id)
            dtse[0][idx] = [density] * self.dtse_shape[2]
            dtse[1][idx] = [flow] * self.dtse_shape[2]
            dtse[2][idx] = [queue_length] * self.dtse_shape[2]
            dtse[3][idx] = [speed] * self.dtse_shape[2]

        return dtse

    def print_dtse(self, dtse):
        print(self.dtse_shape)
        [([print(h) for h in c], print("")) for c in dtse]

    def get_lane_ids_for_edge(self, edge_id):
        try:
            num_lanes = traci.edge.getLaneNumber(edge_id)
            lane_ids = [f"{edge_id}_{i}" for i in range(num_lanes)]
            return lane_ids
        except traci.exceptions.TraCIException as e:
            print(f"Error: Edge '{edge_id}' is not known. Exception: {e}")
            return []


    def get_density(self, edge_id):
        try:
            lane_ids = self.get_lane_ids_for_edge(edge_id)
            if not lane_ids:
                print(f"No lanes found for edge: {edge_id}")
                return 0

            total_vehicles = 0
            total_length = 0
            for lane_id in lane_ids:
                try:
                    total_vehicles += traci.lane.getLastStepVehicleNumber(lane_id)
                    total_length += traci.lane.getLength(lane_id)
                except traci.exceptions.TraCIException:
                    print(f"Warning: Lane '{lane_id}' is not known")
                    continue

            density = total_vehicles / total_length if total_length > 0 else 0
            print(f"Density for edge {edge_id}: {density} (Total Vehicles: {total_vehicles}, Total Length: {total_length})")
            return density
        except Exception as e:
            print(f"Error getting density for edge {edge_id}: {e}")
            return 0

    def get_flow(self, edge_id):
        try:
            lane_ids = self.get_lane_ids_for_edge(edge_id)
            if not lane_ids:
                print(f"No lanes found for edge: {edge_id}")
                return 0

            total_flow = 0
            for lane_id in lane_ids:
                try:
                    total_flow += traci.lane.getLastStepVehicleNumber(lane_id)  # Number of vehicles in the last step
                except traci.exceptions.TraCIException:
                    print(f"Warning: Lane '{lane_id}' is not known")
            return total_flow
        except Exception as e:
            print(f"Error getting flow for edge {edge_id}: {e}")
            return 0



    def get_ramp_queue_length(self):
        queue_length = 0
        for ramp_id, ramp_lanes in self.ramp_lane_mapping.items():
            for lane_id in ramp_lanes:
                try:
                    veh_ids = traci.lane.getLastStepVehicleIDs(lane_id)
                    for veh_id in veh_ids:
                        if traci.vehicle.getSpeed(veh_id) < 0.1:  # Assuming vehicles with speed < 0.1 are queuing
                            queue_length += 1
                except traci.exceptions.TraCIException:
                    print(f"Warning: Lane '{lane_id}' is not known")
        return queue_length

    def get_average_speed(self, edge_id):
        try:
            lane_ids = self.get_lane_ids_for_edge(edge_id)
            if not lane_ids:
                print(f"No lanes found for edge: {edge_id}")
                return 0

            total_speed = 0
            total_vehicles = 0
            for lane_id in lane_ids:
                try:
                    veh_ids = traci.lane.getLastStepVehicleIDs(lane_id)
                    for veh_id in veh_ids:
                        total_speed += traci.vehicle.getSpeed(veh_id)
                        total_vehicles += 1
                except traci.exceptions.TraCIException:
                    print(f"Warning: Lane '{lane_id}' is not known")
            avg_speed = total_speed / total_vehicles if total_vehicles > 0 else 0
            return avg_speed
        except Exception as e:
            print(f"Error getting average speed for edge {edge_id}: {e}")
            return 0


        
    def get_veh_delay(self, veh_id):
        return 1 - (self.get_veh_speed(veh_id) / self.args["v_max_speed"])
    

    
    def get_sum_delay(self, tl_id):
        sum_delay = 0

        for veh_id in self.yield_tl_vehs(tl_id):
            sum_delay += self.get_veh_delay(veh_id)

        return sum_delay
    

    
    def get_sum_waiting_time(self, tl_id):
        sum_waiting_time = 0

        for veh_id in self.yield_tl_vehs(tl_id):
            sum_waiting_time += self.get_veh_waiting_time(veh_id)

        return sum_waiting_time
    

    
    def get_sum_delay_a_sum_waiting_time(self, tl_id):
        sum_delay, sum_waiting_time = 0, 0

        for veh_id in self.yield_tl_vehs(tl_id):
            sum_delay += self.get_veh_delay(veh_id)
            sum_waiting_time += self.get_veh_waiting_time(veh_id)

        return sum_delay, sum_waiting_time
    