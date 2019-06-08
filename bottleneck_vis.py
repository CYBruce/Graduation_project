"""File demonstrating formation of congestion in bottleneck."""

from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, \
    InFlows, SumoLaneChangeParams, SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.core.params import TrafficLightParams

from flow.scenarios.bottleneck import BottleneckScenario
from flow.controllers import SimLaneChangeController, ContinuousRouter
from flow.envs.bottleneck_env import BottleneckEnv, DesiredVelocityEnv
from flow.core.experiment import Experiment

import logging

import numpy as np

import matplotlib.pyplot as plt

import argparse
from datetime import datetime
import gym
import numpy as np
import os
import sys

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune.registry import register_env
from flow.utils.rllib import get_flow_params
from flow.utils.rllib import get_rllib_config
from flow.utils.rllib import get_rllib_pkl
from flow.core.util import emission_to_csv
from flow.utils.registry import make_create_env


SCALING = 1
DISABLE_TB = True
# If set to False, ALINEA will control the ramp meter
DISABLE_RAMP_METER = True
INFLOW = 1700
# congestion happens when inflow over 1700

class BottleneckDensityExperiment(Experiment):

    def __init__(self, env):
        super().__init__(env)

    def run(self, num_runs, num_steps, rl_actions=None, convert_to_csv=False):
        info_dict = {}
        if rl_actions is None:

            def rl_actions(*_):
                return None

        rets = []
        mean_rets = []
        ret_lists = []
        vels = []
        mean_vels = []
        std_vels = []
        mean_densities = []
        mean_outflows = []
        for i in range(num_runs):
            vel = np.zeros(num_steps)
            logging.info('Iter #' + str(i))
            ret = 0
            ret_list = []
            step_outflows = []
            step_densities = []
            state = self.env.reset()
            for j in range(num_steps):
                state, reward, done, _ = self.env.step(rl_actions(state))
                vel[j] = np.mean(self.env.k.vehicle.get_speed(
                    self.env.k.vehicle.get_ids()))
                ret += reward
                ret_list.append(reward)

                env = self.env
                step_outflow = env.k.vehicle.get_outflow_rate(20)
                density = self.env.get_bottleneck_density()

                step_outflows.append(step_outflow)
                step_densities.append(density)
                if done:
                    break
            rets.append(ret)
            vels.append(vel)
            mean_densities.append(sum(step_densities[100:]) /
                                  (num_steps - 100))
            env = self.env
            outflow = env.k.vehicle.get_outflow_rate(10000)
            mean_outflows.append(outflow)
            mean_rets.append(np.mean(ret_list))
            ret_lists.append(ret_list)
            mean_vels.append(np.mean(vel))
            std_vels.append(np.std(vel))
            print('Round {0}, return: {1}'.format(i, ret))

        info_dict['returns'] = rets
        info_dict['velocities'] = vels
        info_dict['mean_returns'] = mean_rets
        info_dict['per_step_returns'] = ret_lists
        info_dict['average_outflow'] = np.mean(mean_outflows)
        info_dict['per_rollout_outflows'] = mean_outflows

        info_dict['average_rollout_density_outflow'] = np.mean(mean_densities)

        print('Average, std return: {}, {}'.format(
            np.mean(rets), np.std(rets)))
        print('Average, std speed: {}, {}'.format(
            np.mean(mean_vels), np.std(std_vels)))
        self.env.terminate()

        return info_dict


def bottleneck_example(flow_rate, horizon, restart_instance=False,
                       render=None):
    """
    Perform a simulation of vehicles on a bottleneck.

    Parameters
    ----------
    flow_rate : float
        total inflow rate of vehicles into the bottleneck
    horizon : int
        time horizon
    restart_instance: bool, optional
        whether to restart the instance upon reset
    render: bool, optional
        specifies whether to use the gui during execution

    Returns
    -------
    exp: flow.core.experiment.Experiment
        A non-rl experiment demonstrating the performance of human-driven
        vehicles on a bottleneck.
    """
    if render is None:
        render = False

    sim_params = SumoParams(
        sim_step=0.5,
        render=render,
        overtake_right=False,
        restart_instance=restart_instance)

    vehicles = VehicleParams()

    vehicles.add(
        veh_id="human",
    	lane_change_controller=(SimLaneChangeController, {}),
    	routing_controller=(ContinuousRouter, {}),
    	car_following_params=SumoCarFollowingParams(
        	speed_mode=31,
    	),
    	lane_change_params=SumoLaneChangeParams(
        	lane_change_mode=00,
        ),
        num_vehicles=1)
    num_observed_segments = [("1", 1), ("2", 3), ("3", 3), ("4", 3), ("5", 1)]
    controlled_segments = [("1", 1, False), ("2", 2, True), ("3", 2, True), ("4", 2, True), ("5", 1, False)]
    additional_env_params = {
    	"target_velocity": 40,
    	"disable_tb": True,
    	"disable_ramp_metering": True,
    	"symmetric": True,
    	"observed_segments": num_observed_segments,
        "controlled_segments": controlled_segments,
    	"reset_inflow": False,
    	"lane_change_duration": 5,
    	"max_accel": 2,
    	"max_decel": 2,
    	"inflow_range": [1000, 2000]
    }
    env_params = EnvParams(
        horizon=horizon, additional_params=additional_env_params)

    inflow = InFlows()
    inflow.add(
        veh_type="human",
        edge="1",
        vehsPerHour=flow_rate,
        departLane="random",
        departSpeed=10)

    traffic_lights = TrafficLightParams()
    if not DISABLE_TB:
        traffic_lights.add(node_id="2")
    if not DISABLE_RAMP_METER:
        traffic_lights.add(node_id="3")

    additional_net_params = {"scaling": SCALING, "speed_limit": 23}
    net_params = NetParams(
        inflows=inflow,
        no_internal_links=False,
        additional_params=additional_net_params)

    initial_config = InitialConfig(
        spacing="random",
        min_gap=5,
        lanes_distribution=float("inf"),
        edges_distribution=["2", "3", "4", "5"])

    scenario = BottleneckScenario(
        name="tcy_base",
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config,
        traffic_lights=traffic_lights)

    env = DesiredVelocityEnv(env_params, sim_params, scenario)

    return BottleneckDensityExperiment(env)


if __name__ == '__main__':
    ray.init(num_cpus = 1)
    result_dir = '/home/user/ray_results/mybottleneckwithoutlane10/PPO_DesiredVelocityEnv-v0_0_2019-03-30_15-47-34wdla6uer/checkpoint_200'
    config = get_rllib_config(result_dir)
    try:
        pkl = get_rllib_pkl(result_dir)
    except Exception:
        pass
    flow_params = get_flow_params(config)
    sim_params = flow_params['sim']
    setattr(sim_params, 'num_clients', 1)

    # Create and register a gym+rllib env
    create_env, env_name = make_create_env(
        params=flow_params, version=0, render=False)
    register_env(env_name, create_env)

    # Determine agent and checkpoint
    config_run = config['env_config']['run'] if 'run' in config['env_config'] \
        else None
    if config_run:
        agent_cls = get_agent_class(config_run)
    else:
        print('visualizer_rllib.py: error: could not find flow parameter '
              '\'run\' in params.json, '
              'add argument --run to provide the algorithm or model used '
              'to train the results\n e.g. '
              'python ./visualizer_rllib.py /tmp/ray/result_dir 1 --run PPO')
        sys.exit(1)

    sim_params.restart_instance = False
    sim_params.emission_path = './test_time_rollout/'
    agent = agent_cls(env=env_name, config=config)
    # import the experiment variable
    # inflow, number of steps, binary
    import pandas as pd
    inflows = list(range(1000,2050,50))
    outflows = []
    returns = []
    mean_returns = []
    per_outflows = []
    rollouts_num = 10
    for inflow in inflows:
        exp = bottleneck_example(inflow, 1500, render=False)
        info_dict = exp.run(rollouts_num, 1500, rl_actions=agent.compute_action())
        returns.append(info_dict['returns'])
        mean_returns.append(info_dict['mean_returns'])
        per_outflows.append(info_dict['per_rollout_outflows'])
        outflows.append(info_dict['average_outflow'])

    plt.figure()
    plt.plot(inflows,outflows)
    plt.show()
    path = '/home/user/桌面/tcy_thesis/base10.csv'
    pd.DataFrame({'INFLOW':inflows, 'OUTFLOW':outflows, 'PER_OUT':per_outflows, 'RETURNS':returns, 'MEAN_RTN':mean_returns}).to_csv(path)





