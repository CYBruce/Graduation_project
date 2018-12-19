from flow.core.params import NetParams, InitialConfig, SumoParams, EnvParams
from flow.scenarios.loop import ADDITIONAL_NET_PARAMS, LoopScenario
from flow.core.vehicles import Vehicles
from flow.controllers import IDMController, ContinuousRouter
from flow.envs.loop.loop_accel import AccelEnv
from flow.core.experiment import SumoExperiment

scenario_name = 'LoopScenario'
name =  'all_human_experient'
net_params = NetParams(additional_params=ADDITIONAL_NET_PARAMS)
initial_config = InitialConfig(spacing='uniform', perturbation=1)
vehicles = Vehicles()
vehicles.add('human', acceleration_controller=(IDMController,{}),
             routing_controller= (ContinuousRouter, {}),
             num_vehicles=22)
sumo_params = SumoParams(sim_step=0.1, render=False)
env_params = EnvParams(
    additional_params={
        'max_accel':3,
        'max_decel':3,
        'target_velocity':10
    },
)

scenario = LoopScenario(
    name = scenario_name,
    vehicles=vehicles,
    net_params = net_params,
    initial_config=initial_config
)
env = AccelEnv(env_params, sumo_params, scenario)
exp = SumoExperiment(env, scenario)
ans = exp.run1(1,6000)
#return a dic(info_dic)

#visualization
velocities = ans['velocities'][0]
velocity = [velocities[i][0] for i in range(6000)]
import  matplotlib.pyplot as plt
plt.figure()
plt.plot(velocity)
plt.show()

