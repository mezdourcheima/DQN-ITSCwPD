# """CHANGE CUSTOM ENV IMPORT HERE""" ##################################################################################
from .custom_env import SUMO_PARAMS, Baselines, RLController 
from .metanet_env import MetaNet   # Import MetaNet model
from .metanet_env.utils import metanet_params
import numpy as np

########################################################################################################################
class DqnEnv:
    
    def min_max_scale(self, x, feature):
        return (x - self.min_max[feature][0]) / (self.min_max[feature][1] - self.min_max[feature][0])

    def __init__(self, m, p=None, traffic_model='sumo', metanet_params=None):
        self.mode = {"train": False, "observe": False, "play": False, m: True}
        self.player = p if self.mode["play"] else None
        self.traffic_model = traffic_model

        # """CHANGE ENV CONSTRUCT HERE""" ##############################################################################
        if self.mode["train"]:
            print(f"I'm using sumo")
            self.sumo_env = RLController(gui=False, log=False, rnd=(True, True))
        elif self.mode["observe"]:
            self.sumo_env = RLController(gui=SUMO_PARAMS["gui"], log=SUMO_PARAMS["log"], rnd=SUMO_PARAMS["rnd"])
        elif self.mode["play"]:
            if p == "Test":
                self.sumo_env = RLController(gui=SUMO_PARAMS["gui"], log=SUMO_PARAMS["log"], rnd=SUMO_PARAMS["rnd"])
            else:
                self.sumo_env = getattr(Baselines, p)(gui=SUMO_PARAMS["gui"], log=SUMO_PARAMS["log"], rnd=SUMO_PARAMS["rnd"])
        ################################################################################################################
        if traffic_model == 'metanet':
                print(f"I'm using metanet")
                self.metanet = MetaNet(params=metanet_params)

        # """CHANGE FEATURE SCALING HERE""" ############################################################################
        self.min_max = {
        }
        ################################################################################################################

        # """CHANGE ACTION AND OBSERVATION SPACE SIZES HERE""" #########################################################
        self.action_space_n = self.sumo_env.action_space_n
        self.observation_space_n = self.sumo_env.observation_space_n
        ################################################################################################################
        self.ramp_edges = self.find_ramp_edges()
        self.edges_after_ramps = self.find_edges_after_ramps()
        ################################################################################################################




    def obs(self):
        # """CHANGE OBSERVATION HERE""" ################################################################################
        obs = self.sumo_env.obs()
        print(f"observations are :{obs}")
        ################################################################################################################
        return obs

    def rew(self):
        # """CHANGE REWARD HERE""" #####################################################################################
        rew = self.sumo_env.rew()
        print(f"reward is : {rew}")
        ################################################################################################################
        return rew

    def done(self):
        # """CHANGE DONE HERE""" #######################################################################################
        done = self.sumo_env.done()
        ################################################################################################################
        return done

    def info(self):
        # """CHANGE INFO HERE""" #######################################################################################
        info = self.sumo_env.log_info()
        #info = self.sumo_env.info() ['l', 'r']
        ################################################################################################################
        return info

    def reset(self):
        # """CHANGE RESET HERE""" ######################################################################################
        if self.traffic_model == 'metanet':
            # Reset MetaNet model
            self.metanet.reset()
        else:
            self.sumo_env.reset()
        ################################################################################################################


    def step(self, action):
            # """CHANGE STEP HERE""" #######################################################################################
        if self.traffic_model == 'metanet':
            densities, flows, speeds = self.get_current_traffic_state()
            controls = self.get_controls(action)
            new_densities, new_flows, new_speeds = self.metanet.update(densities, flows, speeds, controls)
            self.apply_traffic_state(new_densities, new_flows, new_speeds)
        else:
            self.sumo_env.step(action)
        ################################################################################################################

    def reset_render(self):
        # """CHANGE RESET RENDER HERE""" ###############################################################################
        pass
        ################################################################################################################

    def step_render(self):
        # """CHANGE STEP RENDER HERE""" ################################################################################
        pass
        ################################################################################################################

    def log_info(self):
        # """CHANGE log_info HERE""" #######################################################################################
        self.sumo_env.log_info()
    
    def find_ramp_edges(self):
        return self.sumo_env.find_ramp_edges()

    def find_edges_after_ramps(self):
        return self.sumo_env.find_edges_after_ramps()
    
    def get_current_traffic_state(self):
        # Return current densities, flows, and speeds for MetaNet
        return self.metanet.densities, self.metanet.flows, self.metanet.speeds

    def get_controls(self, action):
        # Translate actions into traffic controls for MetaNet
        controls = np.zeros_like(self.densities)  # Simplified for illustration
        # Assuming action is a list of control values corresponding to each segment
        controls = action
        return controls

    def apply_traffic_state(self, densities, flows, speeds):
        # Apply updated traffic state for MetaNet
        self.densities = densities
        self.flows = flows
        self.speeds = speeds