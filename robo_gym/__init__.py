from gym.envs.registration import register

# naming convention: EnvnameRobotSim

# Example Environments
register(
    id='ExampleEnvSim-v0',
    entry_point='robo_gym.envs:ExampleEnvSim',
)

register(
    id='ExampleEnvRob-v0',
    entry_point='robo_gym.envs:ExampleEnvRob',
)

# MiR100 Environments
register(
    id='NoObstacleNavigationMir100Sim-v0',
    entry_point='robo_gym.envs:NoObstacleNavigationMir100Sim',
)

register(
    id='NoObstacleNavigationMir100Rob-v0',
    entry_point='robo_gym.envs:NoObstacleNavigationMir100Rob',
)

register(
    id='ObstacleAvoidanceMir100Sim-v0',
    entry_point='robo_gym.envs:ObstacleAvoidanceMir100Sim',
)

register(
    id='ObstacleAvoidanceMir100Rob-v0',
    entry_point='robo_gym.envs:ObstacleAvoidanceMir100Rob',
)

register(
    id='TrajectoryNavigationMir100Sim-v0',
    entry_point='robo_gym.envs:TrajectoryNavigationMir100Sim',
)

register(
    id='TrajectoryNavigationMir100Rob-v0',
    entry_point='robo_gym.envs:TrajectoryNavigationMir100Rob',
)

# UR Environments
register(
    id='EmptyEnvironmentURSim-v0',
    entry_point='robo_gym.envs:EmptyEnvironmentURSim',
)

register(
    id='EmptyEnvironmentURRob-v0',
    entry_point='robo_gym.envs:EmptyEnvironmentURRob',
)

register(
    id='EndEffectorPositioningURSim-v0',
    entry_point='robo_gym.envs:EndEffectorPositioningURSim',
)

register(
    id='EndEffectorPositioningURRob-v0',
    entry_point='robo_gym.envs:EndEffectorPositioningURRob',
)

register(
    id='BasicAvoidanceURSim-v0',
    entry_point='robo_gym.envs:BasicAvoidanceURSim',
)

register(
    id='BasicAvoidanceURRob-v0',
    entry_point='robo_gym.envs:BasicAvoidanceURRob',
)

register(
    id='AvoidanceRaad2022URSim-v0',
    entry_point='robo_gym.envs:AvoidanceRaad2022URSim',
)

register(
    id='AvoidanceRaad2022URRob-v0',
    entry_point='robo_gym.envs:AvoidanceRaad2022URRob',
)

register(
    id='AvoidanceRaad2022TestURSim-v0',
    entry_point='robo_gym.envs:AvoidanceRaad2022TestURSim',
)

register(
    id='AvoidanceRaad2022TestURRob-v0',
    entry_point='robo_gym.envs:AvoidanceRaad2022TestURRob',
)




