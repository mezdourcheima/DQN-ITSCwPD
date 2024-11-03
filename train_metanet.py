from env import HYPER_PARAMS_METANET, network_config_metanet
from dqn import CustomEnvWrapper, make_env, Agents
from env.metanet_env import MetaNetEnv
from env import DqnEnvMetaNet  # Assuming you named the modified class file as dqn_env_metanet.py
import pickle
import torch as T
import torch
import os
import time
import argparse
import itertools
from datetime import timedelta



class Train:
    def __init__(self, args):

        # Set up device for M1
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using device: MPS (Metal Performance Shaders)")
        else:
            self.device = torch.device("cpu")
            print("Using device: CPU")


        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        

        # Initialize DqnEnv with MetaNetEnv
        env = CustomEnvWrapper(DqnEnvMetaNet(type(self).__name__.lower()))  # 'train' mode, you can adjust based on your needs

        self.env = make_env(
            env=env,
            repeat=args.repeat,
            max_episode_steps=args.max_episode_steps,
            n_env=args.n_env
        )
        input_dim = self.env.observation_space  # This should now be correctly formatted as a tuple of integers
        output_dim = self.env.action_space.n
        # Initialize the RL agent with the environment's observation and action spaces
        self.agent = getattr(Agents, args.algo)(
            n_env=args.n_env,
            lr=args.lr,
            gamma=args.gamma,
            epsilon_start=args.eps_start,
            epsilon_min=args.eps_min,
            epsilon_decay=args.eps_dec,
            epsilon_exp_decay=args.eps_dec_exp,
            nn_conf_func=network_config_metanet,
            input_dim=input_dim,  # Assuming a 1D observation space
            output_dim=output_dim,  # For discrete actions
            batch_size=args.bs,
            min_buffer_size=args.min_mem,
            buffer_size=args.max_mem,
            update_target_frequency=args.target_update_freq,
            target_soft_update=args.target_soft_update,
            target_soft_update_tau=args.target_soft_update_tau,
            save_frequency=args.save_freq,
            log_frequency=args.log_freq,
            save_dir=args.save_dir,
            log_dir=args.log_dir,
            load=args.load,
            algo=args.algo,
            gpu=args.gpu,

        )

        self.agent.online_network.to(self.device)
        self.agent.target_network.to(self.device)  # If you have a target network, move it to the device too
        self.agent.load_model()

        print()
        print("TRAIN")
        print()
        print(args.algo)
        print()
        print(self.agent.online_network)
        print()
        [print(arg, "=", getattr(args, arg)) for arg in vars(args)]

        self.max_total_steps = args.max_total_steps

        
    def init_replay_memory_buffer(self):
        print()
        print("Initialize Replay Memory Buffer")

        obses = self.env.reset()
        for t in range(self.agent.min_buffer_size // self.agent.n_env):
            if t >= (self.agent.min_buffer_size // self.agent.n_env) - self.agent.resume_step:
                actions = self.agent.choose_actions(obses)
            else:
                actions = [self.env.action_space.sample() for _ in range(self.agent.n_env)]

            new_obses, rews, dones, _ = self.env.step(actions)
            self.agent.store_transitions(obses, actions, rews, dones, new_obses, None)

            obses = new_obses

            if (t+1) % (10000 // self.agent.n_env) == 0:
                print(str((t+1) * self.agent.n_env) + ' / ' + str(self.agent.min_buffer_size))
                print('---', str(timedelta(seconds=round((time.time() - self.agent.start_time), 0))), '---')

    def process_data(self, data):
            # This function moves tensors to the correct device if needed
            if isinstance(data, T.Tensor):
                return data.to(self.device)
            elif isinstance(data, (list, tuple)):
                return [self.process_data(d) for d in data]
            else:
                return data
            
    
    
    def train_loop(self):
        print()
        print("Start Training")

        obses = self.env.reset()
        obses = self.process_data(obses)
        print(f"observations:{obses}")


        for step in itertools.count(start=self.agent.resume_step):
                self.agent.step = step

                actions = self.agent.choose_actions(obses)

                new_obses, rews, dones, infos = self.env.step(actions)
                new_obses, rews, dones, infos = map(self.process_data, [new_obses, rews, dones, infos])
                print(f"rewards:{rews}")
                print(f"dones:{dones}")
                print(f"Infos:{infos}")
                

                
                self.agent.store_transitions(obses, actions, rews, dones, new_obses, infos)

                obses = new_obses

                self.agent.learn()

                self.agent.update_target_network()

                self.agent.log()

                self.agent.save_model()


               
                if bool(self.max_total_steps) and (step * self.agent.n_env) >= self.max_total_steps:
                    exit()
        

    
        
    def run(self):
        self.init_replay_memory_buffer()
        self.train_loop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRAIN")
    str2bool = (lambda v: v.lower() in ("yes", "y", "true", "t", "1"))
    parser.add_argument('-gpu', type=str, default=HYPER_PARAMS_METANET["gpu"], help='GPU #')
    parser.add_argument('-n_env', type=int, default=HYPER_PARAMS_METANET["n_env"], help='Multi-processing environments')
    parser.add_argument('-lr', type=float, default=HYPER_PARAMS_METANET["lr"], help='Learning rate')
    parser.add_argument('-gamma', type=float, default=HYPER_PARAMS_METANET["gamma"], help='Discount factor')
    parser.add_argument('-eps_start', type=float, default=HYPER_PARAMS_METANET["eps_start"], help='Epsilon start')
    parser.add_argument('-eps_min', type=float, default=HYPER_PARAMS_METANET["eps_min"], help='Epsilon min')
    parser.add_argument('-eps_dec', type=float, default=HYPER_PARAMS_METANET["eps_dec"], help='Epsilon decay')
    parser.add_argument('-eps_dec_exp', type=str2bool, default=HYPER_PARAMS_METANET["eps_dec_exp"], help='Epsilon exponential decay')
    parser.add_argument('-bs', type=int, default=HYPER_PARAMS_METANET["bs"], help='Batch size')
    parser.add_argument('-min_mem', type=int, default=HYPER_PARAMS_METANET["min_mem"], help='Replay memory buffer min size')
    parser.add_argument('-max_mem', type=int, default=HYPER_PARAMS_METANET["max_mem"], help='Replay memory buffer max size')
    parser.add_argument('-target_update_freq', type=int, default=HYPER_PARAMS_METANET["target_update_freq"], help='Target network update frequency')
    parser.add_argument('-target_soft_update', type=str2bool, default=HYPER_PARAMS_METANET["target_soft_update"], help='Target network soft update')
    parser.add_argument('-target_soft_update_tau', type=float, default=HYPER_PARAMS_METANET["target_soft_update_tau"], help='Target network soft update tau rate')
    parser.add_argument('-save_freq', type=int, default=HYPER_PARAMS_METANET["save_freq"], help='Save frequency')
    parser.add_argument('-log_freq', type=int, default=HYPER_PARAMS_METANET["log_freq"], help='Log frequency')
    parser.add_argument('-save_dir', type=str, default=HYPER_PARAMS_METANET["save_dir"], help='Save directory')
    parser.add_argument('-log_dir', type=str, default=HYPER_PARAMS_METANET["log_dir"], help='Log directory')
    parser.add_argument('-load', type=str2bool, default=HYPER_PARAMS_METANET["load"], help='Load model')
    parser.add_argument('-repeat', type=int, default=HYPER_PARAMS_METANET["repeat"], help='Steps repeat action')
    parser.add_argument('-max_episode_steps', type=int, default=HYPER_PARAMS_METANET["max_episode_steps"], help='Episode step limit')
    parser.add_argument('-max_total_steps', type=int, default=HYPER_PARAMS_METANET["max_total_steps"], help='Max total training steps')
    parser.add_argument('-algo', type=str, default=HYPER_PARAMS_METANET["algo"],
                        help='DQNAgent ' +
                             'DoubleDQNAgent ' +
                             'DuelingDoubleDQNAgent ' +
                             'PerDuelingDoubleDQNAgent'
                        )

    Train(parser.parse_args()).run()
