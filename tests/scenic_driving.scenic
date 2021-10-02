import scenic

param map = localPath('Town01.xodr')
param carla_map = 'Town01'

model scenic.domains.driving.model

simulator scenic.core.simulators.DummySimulator()

ego = Car at Range(-5, 5) @ 2, with behavior FollowLaneBehavior
