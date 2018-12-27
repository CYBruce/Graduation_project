"""Visualizer for rllib experiments.

Attributes
----------
EXAMPLE_USAGE : str
    Example call to the function, which is
    ::

        python ./visualizer_rllib.py /tmp/ray/result_dir 1

parser : ArgumentParser
    Command-line argument parser
"""

import argparse
from datetime import datetime
import numpy as np
import os
import sys

import ray
from ray.rllib.agents.agent import get_agent_class
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

from flow.core.util import emission_to_csv
from flow.utils.registry import make_create_env
from flow.utils.rllib import get_flow_params
from flow.utils.rllib import get_rllib_config
from flow.utils.rllib import get_rllib_pkl


EXAMPLE_USAGE = """
example usage:
    python ./visualizer_rllib.py /tmp/ray/result_dir 1

Here the arguments are:
1 - the number of the checkpoint
"""


def visualizer_rllib(args):
    result_dir = args.result_dir if args.result_dir[-1] != '/' \
        else args.result_dir[:-1]

    # config = get_rllib_config(result_dir + '/..')
    # pkl = get_rllib_pkl(result_dir + '/..')
    config = get_rllib_config(result_dir)
    # TODO(ev) backwards compatibility hack
    try:
        pkl = get_rllib_pkl(result_dir)
    except Exception:
        pass

    # check if we have a multiagent scenario but in a
    # backwards compatible way
    if config.get('multiagent', {}).get('policy_graphs', {}):
        multiagent = True
        config['multiagent'] = pkl['multiagent']
    else:
        multiagent = False

    # Run on only one cpu for rendering purposes
    config['num_workers'] = 0

    flow_params = get_flow_params(config)

    # hack for old pkl files
    # TODO(ev) remove eventually
    sumo_params = flow_params['sumo']
    setattr(sumo_params, 'num_clients', 1)

    # Create and register a gym+rllib env
    create_env, env_name = make_create_env(
        params=flow_params, version=0, render=False)
    register_env(env_name, create_env)

    # Determine agent and checkpoint
    config_run = config['env_config']['run'] if 'run' in config['env_config'] \
        else None
    if (args.run and config_run):
        if (args.run != config_run):
            print('visualizer_rllib.py: error: run argument '
                  + '\'{}\' passed in '.format(args.run)
                  + 'differs from the one stored in params.json '
                  + '\'{}\''.format(config_run))
            sys.exit(1)
    if (args.run):
        agent_cls = get_agent_class(args.run)
    elif (config_run):
        agent_cls = get_agent_class(config_run)
    else:
        print('visualizer_rllib.py: error: could not find flow parameter '
              '\'run\' in params.json, '
              'add argument --run to provide the algorithm or model used '
              'to train the results\n e.g. '
              'python ./visualizer_rllib.py /tmp/ray/result_dir 1 --run PPO')
        sys.exit(1)

    sumo_params.restart_instance = False

    sumo_params.emission_path = './test_time_rollout/'

    # pick your rendering mode
    if args.render_mode == 'sumo-web3d':
        sumo_params.num_clients = 2
        sumo_params.render = False
    elif args.render_mode == 'drgb':
        sumo_params.render = 'drgb'
        sumo_params.pxpm = 4
    elif args.render_mode == 'sumo-gui':
        sumo_params.render = False
    elif args.render_mode == 'no-render':
        sumo_params.render = False

    if args.save_render:
        sumo_params.render = 'drgb'
        sumo_params.pxpm = 4
        sumo_params.save_render = True

    # Recreate the scenario from the pickled parameters
    exp_tag = flow_params['exp_tag']
    net_params = flow_params['net']
    vehicles = flow_params['veh']
    initial_config = flow_params['initial']
    module = __import__('flow.scenarios', fromlist=[flow_params['scenario']])
    scenario_class = getattr(module, flow_params['scenario'])

    scenario = scenario_class(
        name=exp_tag,
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config)

    # Start the environment with the gui turned on and a path for the
    # emission file
    module = __import__('flow.envs', fromlist=[flow_params['env_name']])
    env_class = getattr(module, flow_params['env_name'])
    env_params = flow_params['env']
    env_params.restart_instance = False
    if args.evaluate:
        env_params.evaluate = True

    # create the agent that will be used to compute the actions
    agent = agent_cls(env=env_name, config=config)
    checkpoint = result_dir + '/checkpoint_' + args.checkpoint_num
    checkpoint = checkpoint + '/checkpoint-' + args.checkpoint_num
    agent.restore(checkpoint)

    env = ModelCatalog.get_preprocessor_as_wrapper(env_class(
        env_params=env_params, sumo_params=sumo_params, scenario=scenario))

    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    fig = plt.figure()
    h = np.linspace(0, 60, 100)
    Deltav = np.linspace(-6, 12, 100)
    Headway, DELTAV = np.meshgrid(h, Deltav)
    # fix v=20m/s
    xn, yn = Headway.shape
    geta = np.array(Headway)
    for xk in range(xn):
        for yk in range(yn):
            #输入状态
            #Headway[xk,yk]
            #DELTAV[xk,yk]
            geta[xk, yk] = agent.compute_action(np.array([3.8/30,DELTAV[xk,yk]/30,Headway[xk,yk]/260]))
    surf = plt.contourf(DELTAV, Headway, geta, 20, cmap=cm.coolwarm)
    plt.colorbar()
    #C = plt.contour(DELTAV, Headway, geta, 20, colors='black')
    # plt.clabel(C, inline = True, fontsize = 10)
    plt.show()
    # terminate the environment
    env.unwrapped.terminate()

    # if prompted, convert the emission file into a csv file
    if args.emission_to_csv:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        emission_filename = '{0}-emission.xml'.format(scenario.name)

        emission_path = \
            '{0}/test_time_rollout/{1}'.format(dir_path, emission_filename)

        emission_to_csv(emission_path)

    # if we wanted to save the render, here we create the movie
    '''
    if args.save_render:
        dirs = os.listdir(os.path.expanduser('~')+'/flow_rendering')
        dirs.sort(key=lambda date: datetime.strptime(date, "%Y-%m-%d-%H%M%S"))
        recent_dir = dirs[-1]
        # create the movie
        movie_dir = os.path.expanduser('~') + '/flow_rendering/' + recent_dir
        save_dir = os.path.expanduser('~') + '/flow_movies'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        os_cmd = "cd " + movie_dir + " && ffmpeg -i frame_%06d.png"
        os_cmd += " -pix_fmt yuv420p " + dirs[-1] + ".mp4"
        os_cmd += "&& cp " + dirs[-1] + ".mp4 " + save_dir + "/"
        os.system(os_cmd)
    '''

def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='[Flow] Evaluates a reinforcement learning agent '
                    'given a checkpoint.',
        epilog=EXAMPLE_USAGE)

    # required input parameters
    parser.add_argument(
        'result_dir', type=str, help='Directory containing results')
    parser.add_argument('checkpoint_num', type=str, help='Checkpoint number.')

    # optional input parameters
    parser.add_argument(
        '--run',
        type=str,
        help='The algorithm or model to train. This may refer to '
             'the name of a built-on algorithm (e.g. RLLib\'s DQN '
             'or PPO), or a user-defined trainable function or '
             'class registered in the tune registry. '
             'Required for results trained with flow-0.2.0 and before.')
    parser.add_argument(
        '--num-rollouts',
        type=int,
        default=1,
        help='The number of rollouts to visualize.')
    parser.add_argument(
        '--emission-to-csv',
        action='store_true',
        help='Specifies whether to convert the emission file '
             'created by sumo into a csv file')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Specifies whether to use the \'evaluate\' reward '
             'for the environment.')
    parser.add_argument(
        '--render-mode',
        type=str,
        default='sumo-gui',
        help='Pick the render mode. Options include sumo-web3d, '
             'rgbd, sumo-gui, and no-render. For more details'
             'see the visualization tutorial.')
    parser.add_argument(
        '--save_render',
        action='store_true',
        help='saves the render to a file')
    parser.add_argument(
        '--horizon',
        type=int,
        help='Specifies the horizon.')
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    ray.init(num_cpus=1)
    visualizer_rllib(args)
