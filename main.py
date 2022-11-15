import os
import time
import shutil


class DroneEnv1:
    def __init__(self, is_save_img_bbox_detection=False, is_save_data_suction_apple=False):
        from drone_tensorforce_env1 import DroneTensorforceEnv1
        self.drone_env1 = DroneTensorforceEnv1(is_save_img_bbox_detection=is_save_img_bbox_detection,
                                               is_save_data_suction_apple=is_save_data_suction_apple)

    def train_network(self, network, path_to_save, epochs=300, agent='ppo', learning_rate=0.0001, is_pretrain=False,
                      path_pretrain='data/env1/recorder_manual_system', max_files_pretrain=100):
        self.drone_env1.set_exploration(type_exploration=801, current_exploration=0.35, min_exploration=0.35,
                                        decay_exploration=0.0001, ignore_exploration_each_n_epoch=5)
        self.drone_env1.create_agent(network=network, agent=agent, learning_rate=learning_rate)
        if is_pretrain:
            self.drone_env1.pretrain(path_pretrain, max_files_pretrain=max_files_pretrain)
        shutil.rmtree(path_to_save, ignore_errors=True)
        os.mkdir(path_to_save)
        self.drone_env1.run(epochs=epochs, path_backup_checkpoint=path_to_save, path_csv=f'{path_to_save}/historical.csv')

    def valid_network(self, epochs=20, path_csv='data/backup/historical.csv', test_suction=False, use_tracker=False,
                      test_tracker=False, dist_y_between_apple=0.075, dist_z_between_apple=0.08):
        self.drone_env1.set_exploration(type_exploration=801, current_exploration=0.0, min_exploration=0.35,
                                        decay_exploration=0.0001, ignore_exploration_each_n_epoch=5)
        self.drone_env1.load_agent()
        self.drone_env1.run(epochs=epochs, path_csv=path_csv, do_update=False, save_best=False,
                            test_suction=test_suction, use_tracker=use_tracker, test_tracker=test_tracker,
                            dist_y_between_apple=dist_y_between_apple, dist_z_between_apple=dist_z_between_apple)

    def create_manual_system_data(self, epochs=100, path_recorder='data/env1/recorder_manual_system'):
        self.drone_env1.create_agent()
        self.drone_env1.run(epochs=epochs, do_update=False, manual_system=True, save_best=False,
                            path_recorder=path_recorder, path_csv=f'{path_recorder}/historical.csv')


class DroneEnv2:
    def __init__(self):
        from drone_tensorforce_env2 import DroneTensorforceEnv2
        self.drone_env2 = DroneTensorforceEnv2()

    def valid_network(self, path_csv, epochs=400):
        self.drone_env2.run(epochs=epochs, path_csv=path_csv)


# ##### Environment 1 #####
# 1 == Create manual system data
# 2 == Train PPO and DDPG
# 22 == Valid all results of PPO and DDPG
# 3 == Train Architecture 1, 2 and 3
# 33 == Valid all results of Architecture 1, 2 and 3
# 4 == Valid tracker
# 5 == Collect data to train suction (the agent that will be executed must be in the path "components/component3/agent")
# 55 == Valid suction

# ##### Environment 2 #####
# 6 == Valid Dronapple

what_to_run = 6

# ### Create manual system data to pretrain the second movement test networks ###
if what_to_run == 1:
    drone_env1 = DroneEnv1()
    drone_env1.create_manual_system_data()

# ########################################## First movement test ##########################################
if what_to_run == 2:
    drone_env1 = DroneEnv1()
    drone_env1.train_network('auto', 'data/env1/Comp3_movement_first/test1', agent='ddpg', learning_rate=0.0001,
                             is_pretrain=True)
    drone_env1.train_network('auto', 'data/env1/Comp3_movement_first/test2', agent='ppo', learning_rate=0.0001,
                             is_pretrain=True)

if what_to_run == 22:
    drone_env1 = DroneEnv1()
    path_folders = 'data/env1/Comp3_movement_first'
    #folders_objective = ['test4']
    for main_folder in os.listdir(path_folders):
        if os.path.isdir(f'{path_folders}/{main_folder}'):
            #if main_folder in folders_objective:
            for checkpoint_folder in os.listdir(f'{path_folders}/{main_folder}'):
                if os.path.isdir(f'{path_folders}/{main_folder}/{checkpoint_folder}'):
                    print(f'{path_folders}/{main_folder}/{checkpoint_folder}')
                    shutil.rmtree('data/agent_checkpoints')
                    time.sleep(0.1)
                    shutil.copytree(f'{path_folders}/{main_folder}/{checkpoint_folder}', 'data/agent_checkpoints')
                    drone_env1.valid_network(path_csv=f'{path_folders}/{main_folder}/{checkpoint_folder}.csv')
###########################################################################################################

# ########################################## Second movement test ##########################################
if what_to_run == 3:
    drone_env1 = DroneEnv1()

    # ########################################## Architecture 1 ##########################################
    drone_env1.train_network('auto', 'data/env1/Comp3_movement_second/test1', is_pretrain=True)
    
    custom_network = [
                [
                    dict(type='retrieve', name='bbox_retrieve', tensors=['bbox']),
                    dict(type='dense', name='bbox_dense0', size=64, bias=True, activation='relu'),
                    dict(type='dense', name='bbox_dense1', size=64, bias=True, activation='relu'),
                    dict(type='register', name='bbox_register', tensor='bbox-embedding')
                ],
                [
                    dict(type='retrieve', name='velocity_retrieve', tensors=['velocity']),
                    dict(type='dense', name='velocity_dense0', size=64, bias=True, activation='relu'),
                    dict(type='dense', name='velocity_dense1', size=64, bias=True, activation='relu'),
                    dict(type='register', name='velocity_register', tensor='velocity-embedding')
                ],
                [
                    dict(type='retrieve', tensors=['bbox-embedding', 'velocity-embedding'], aggregation='concat'),
                    dict(type='dense', name='dense0', size=64, bias=True, activation='relu'),
                ]
            ]
    drone_env1.train_network(custom_network, 'data/env1/Comp3_movement_second/test2', is_pretrain=True)

    '''
    # Error
    custom_network = [
                [
                    dict(type='retrieve', name='bbox_retrieve', tensors=['bbox']),
                    dict(type='dense', name='bbox_dense0', size=64, bias=True, activation='crelu'),
                    dict(type='dense', name='bbox_dense1', size=64, bias=True, activation='crelu'),
                    dict(type='register', name='bbox_register', tensor='bbox-embedding')
                ],
                [
                    dict(type='retrieve', name='velocity_retrieve', tensors=['velocity']),
                    dict(type='dense', name='velocity_dense0', size=64, bias=True, activation='crelu'),
                    dict(type='dense', name='velocity_dense1', size=64, bias=True, activation='crelu'),
                    dict(type='register', name='velocity_register', tensor='velocity-embedding')
                ],
                [
                    dict(type='retrieve', tensors=['bbox-embedding', 'velocity-embedding'], aggregation='concat'),
                    dict(type='dense', name='dense0', size=64, bias=True, activation='crelu'),
                ]
            ]
    #drone_env1.train_network(custom_network, 'data/env1/Comp3_movement_second/test3', is_pretrain=True)
    '''

    custom_network = [
                [
                    dict(type='retrieve', name='bbox_retrieve', tensors=['bbox']),
                    dict(type='dense', name='bbox_dense0', size=64, bias=True, activation='elu'),
                    dict(type='dense', name='bbox_dense1', size=64, bias=True, activation='elu'),
                    dict(type='register', name='bbox_register', tensor='bbox-embedding')
                ],
                [
                    dict(type='retrieve', name='velocity_retrieve', tensors=['velocity']),
                    dict(type='dense', name='velocity_dense0', size=64, bias=True, activation='elu'),
                    dict(type='dense', name='velocity_dense1', size=64, bias=True, activation='elu'),
                    dict(type='register', name='velocity_register', tensor='velocity-embedding')
                ],
                [
                    dict(type='retrieve', tensors=['bbox-embedding', 'velocity-embedding'], aggregation='concat'),
                    dict(type='dense', name='dense0', size=64, bias=True, activation='elu'),
                ]
            ]
    drone_env1.train_network(custom_network, 'data/env1/Comp3_movement_second/test4', is_pretrain=True)

    custom_network = [
                [
                    dict(type='retrieve', name='bbox_retrieve', tensors=['bbox']),
                    dict(type='dense', name='bbox_dense0', size=64, bias=True, activation='leaky-relu'),
                    dict(type='dense', name='bbox_dense1', size=64, bias=True, activation='leaky-relu'),
                    dict(type='register', name='bbox_register', tensor='bbox-embedding')
                ],
                [
                    dict(type='retrieve', name='velocity_retrieve', tensors=['velocity']),
                    dict(type='dense', name='velocity_dense0', size=64, bias=True, activation='leaky-relu'),
                    dict(type='dense', name='velocity_dense1', size=64, bias=True, activation='leaky-relu'),
                    dict(type='register', name='velocity_register', tensor='velocity-embedding')
                ],
                [
                    dict(type='retrieve', tensors=['bbox-embedding', 'velocity-embedding'], aggregation='concat'),
                    dict(type='dense', name='dense0', size=64, bias=True, activation='leaky-relu'),
                ]
            ]
    drone_env1.train_network(custom_network, 'data/env1/Comp3_movement_second/test5', is_pretrain=True)

    custom_network = [
                [
                    dict(type='retrieve', name='bbox_retrieve', tensors=['bbox']),
                    dict(type='dense', name='bbox_dense0', size=64, bias=True, activation='none'),
                    dict(type='dense', name='bbox_dense1', size=64, bias=True, activation='none'),
                    dict(type='register', name='bbox_register', tensor='bbox-embedding')
                ],
                [
                    dict(type='retrieve', name='velocity_retrieve', tensors=['velocity']),
                    dict(type='dense', name='velocity_dense0', size=64, bias=True, activation='none'),
                    dict(type='dense', name='velocity_dense1', size=64, bias=True, activation='none'),
                    dict(type='register', name='velocity_register', tensor='velocity-embedding')
                ],
                [
                    dict(type='retrieve', tensors=['bbox-embedding', 'velocity-embedding'], aggregation='concat'),
                    dict(type='dense', name='dense0', size=64, bias=True, activation='none'),
                ]
            ]
    drone_env1.train_network(custom_network, 'data/env1/Comp3_movement_second/test6', is_pretrain=True)

    custom_network = [
                [
                    dict(type='retrieve', name='bbox_retrieve', tensors=['bbox']),
                    dict(type='dense', name='bbox_dense0', size=64, bias=True, activation='selu'),
                    dict(type='dense', name='bbox_dense1', size=64, bias=True, activation='selu'),
                    dict(type='register', name='bbox_register', tensor='bbox-embedding')
                ],
                [
                    dict(type='retrieve', name='velocity_retrieve', tensors=['velocity']),
                    dict(type='dense', name='velocity_dense0', size=64, bias=True, activation='selu'),
                    dict(type='dense', name='velocity_dense1', size=64, bias=True, activation='selu'),
                    dict(type='register', name='velocity_register', tensor='velocity-embedding')
                ],
                [
                    dict(type='retrieve', tensors=['bbox-embedding', 'velocity-embedding'], aggregation='concat'),
                    dict(type='dense', name='dense0', size=64, bias=True, activation='selu'),
                ]
            ]
    drone_env1.train_network(custom_network, 'data/env1/Comp3_movement_second/test7', is_pretrain=True)

    custom_network = [
                [
                    dict(type='retrieve', name='bbox_retrieve', tensors=['bbox']),
                    dict(type='dense', name='bbox_dense0', size=64, bias=True, activation='sigmoid'),
                    dict(type='dense', name='bbox_dense1', size=64, bias=True, activation='sigmoid'),
                    dict(type='register', name='bbox_register', tensor='bbox-embedding')
                ],
                [
                    dict(type='retrieve', name='velocity_retrieve', tensors=['velocity']),
                    dict(type='dense', name='velocity_dense0', size=64, bias=True, activation='sigmoid'),
                    dict(type='dense', name='velocity_dense1', size=64, bias=True, activation='sigmoid'),
                    dict(type='register', name='velocity_register', tensor='velocity-embedding')
                ],
                [
                    dict(type='retrieve', tensors=['bbox-embedding', 'velocity-embedding'], aggregation='concat'),
                    dict(type='dense', name='dense0', size=64, bias=True, activation='sigmoid'),
                ]
            ]
    drone_env1.train_network(custom_network, 'data/env1/Comp3_movement_second/test8', is_pretrain=True)

    # ########################################## Architecture 2 ##########################################
    custom_network = [
                [
                    dict(type='retrieve', name='bbox_retrieve', tensors=['bbox']),
                    dict(type='dense', name='bbox_dense0', size=128, bias=True, activation='tanh'),
                    dict(type='dense', name='bbox_dense1', size=128, bias=True, activation='tanh'),
                    dict(type='register', name='bbox_register', tensor='bbox-embedding')
                ],
                [
                    dict(type='retrieve', name='velocity_retrieve', tensors=['velocity']),
                    dict(type='dense', name='velocity_dense0', size=128, bias=True, activation='tanh'),
                    dict(type='dense', name='velocity_dense1', size=128, bias=True, activation='tanh'),
                    dict(type='register', name='velocity_register', tensor='velocity-embedding')
                ],
                [
                    dict(type='retrieve', tensors=['bbox-embedding', 'velocity-embedding'], aggregation='concat'),
                    dict(type='dense', name='dense0', size=128, bias=True, activation='tanh'),
                ]
            ]
    drone_env1.train_network(custom_network, 'data/env1/Comp3_movement_second/test9', is_pretrain=True)
    custom_network = [
                [
                    dict(type='retrieve', name='bbox_retrieve', tensors=['bbox']),
                    dict(type='dense', name='bbox_dense0', size=128, bias=True, activation='relu'),
                    dict(type='dense', name='bbox_dense1', size=128, bias=True, activation='relu'),
                    dict(type='register', name='bbox_register', tensor='bbox-embedding')
                ],
                [
                    dict(type='retrieve', name='velocity_retrieve', tensors=['velocity']),
                    dict(type='dense', name='velocity_dense0', size=128, bias=True, activation='relu'),
                    dict(type='dense', name='velocity_dense1', size=128, bias=True, activation='relu'),
                    dict(type='register', name='velocity_register', tensor='velocity-embedding')
                ],
                [
                    dict(type='retrieve', tensors=['bbox-embedding', 'velocity-embedding'], aggregation='concat'),
                    dict(type='dense', name='dense0', size=128, bias=True, activation='relu'),
                ]
            ]
    drone_env1.train_network(custom_network, 'data/env1/Comp3_movement_second/test10', is_pretrain=True)

    '''
    # Error
    custom_network = [
                [
                    dict(type='retrieve', name='bbox_retrieve', tensors=['bbox']),
                    dict(type='dense', name='bbox_dense0', size=128, bias=True, activation='crelu'),
                    dict(type='dense', name='bbox_dense1', size=128, bias=True, activation='crelu'),
                    dict(type='register', name='bbox_register', tensor='bbox-embedding')
                ],
                [
                    dict(type='retrieve', name='velocity_retrieve', tensors=['velocity']),
                    dict(type='dense', name='velocity_dense0', size=128, bias=True, activation='crelu'),
                    dict(type='dense', name='velocity_dense1', size=128, bias=True, activation='crelu'),
                    dict(type='register', name='velocity_register', tensor='velocity-embedding')
                ],
                [
                    dict(type='retrieve', tensors=['bbox-embedding', 'velocity-embedding'], aggregation='concat'),
                    dict(type='dense', name='dense0', size=128, bias=True, activation='crelu'),
                ]
            ]
    #drone_env1.train_network(custom_network, 'data/env1/Comp3_movement_second/test11', is_pretrain=True)
    '''

    custom_network = [
                [
                    dict(type='retrieve', name='bbox_retrieve', tensors=['bbox']),
                    dict(type='dense', name='bbox_dense0', size=128, bias=True, activation='elu'),
                    dict(type='dense', name='bbox_dense1', size=128, bias=True, activation='elu'),
                    dict(type='register', name='bbox_register', tensor='bbox-embedding')
                ],
                [
                    dict(type='retrieve', name='velocity_retrieve', tensors=['velocity']),
                    dict(type='dense', name='velocity_dense0', size=128, bias=True, activation='elu'),
                    dict(type='dense', name='velocity_dense1', size=128, bias=True, activation='elu'),
                    dict(type='register', name='velocity_register', tensor='velocity-embedding')
                ],
                [
                    dict(type='retrieve', tensors=['bbox-embedding', 'velocity-embedding'], aggregation='concat'),
                    dict(type='dense', name='dense0', size=128, bias=True, activation='elu'),
                ]
            ]
    drone_env1.train_network(custom_network, 'data/env1/Comp3_movement_second/test12', is_pretrain=True)

    custom_network = [
                [
                    dict(type='retrieve', name='bbox_retrieve', tensors=['bbox']),
                    dict(type='dense', name='bbox_dense0', size=128, bias=True, activation='leaky-relu'),
                    dict(type='dense', name='bbox_dense1', size=128, bias=True, activation='leaky-relu'),
                    dict(type='register', name='bbox_register', tensor='bbox-embedding')
                ],
                [
                    dict(type='retrieve', name='velocity_retrieve', tensors=['velocity']),
                    dict(type='dense', name='velocity_dense0', size=128, bias=True, activation='leaky-relu'),
                    dict(type='dense', name='velocity_dense1', size=128, bias=True, activation='leaky-relu'),
                    dict(type='register', name='velocity_register', tensor='velocity-embedding')
                ],
                [
                    dict(type='retrieve', tensors=['bbox-embedding', 'velocity-embedding'], aggregation='concat'),
                    dict(type='dense', name='dense0', size=128, bias=True, activation='leaky-relu'),
                ]
            ]
    drone_env1.train_network(custom_network, 'data/env1/Comp3_movement_second/test13', is_pretrain=True)

    custom_network = [
                [
                    dict(type='retrieve', name='bbox_retrieve', tensors=['bbox']),
                    dict(type='dense', name='bbox_dense0', size=128, bias=True, activation='none'),
                    dict(type='dense', name='bbox_dense1', size=128, bias=True, activation='none'),
                    dict(type='register', name='bbox_register', tensor='bbox-embedding')
                ],
                [
                    dict(type='retrieve', name='velocity_retrieve', tensors=['velocity']),
                    dict(type='dense', name='velocity_dense0', size=128, bias=True, activation='none'),
                    dict(type='dense', name='velocity_dense1', size=128, bias=True, activation='none'),
                    dict(type='register', name='velocity_register', tensor='velocity-embedding')
                ],
                [
                    dict(type='retrieve', tensors=['bbox-embedding', 'velocity-embedding'], aggregation='concat'),
                    dict(type='dense', name='dense0', size=128, bias=True, activation='none'),
                ]
            ]
    drone_env1.train_network(custom_network, 'data/env1/Comp3_movement_second/test14', is_pretrain=True)

    custom_network = [
                [
                    dict(type='retrieve', name='bbox_retrieve', tensors=['bbox']),
                    dict(type='dense', name='bbox_dense0', size=128, bias=True, activation='selu'),
                    dict(type='dense', name='bbox_dense1', size=128, bias=True, activation='selu'),
                    dict(type='register', name='bbox_register', tensor='bbox-embedding')
                ],
                [
                    dict(type='retrieve', name='velocity_retrieve', tensors=['velocity']),
                    dict(type='dense', name='velocity_dense0', size=128, bias=True, activation='selu'),
                    dict(type='dense', name='velocity_dense1', size=128, bias=True, activation='selu'),
                    dict(type='register', name='velocity_register', tensor='velocity-embedding')
                ],
                [
                    dict(type='retrieve', tensors=['bbox-embedding', 'velocity-embedding'], aggregation='concat'),
                    dict(type='dense', name='dense0', size=128, bias=True, activation='selu'),
                ]
            ]
    drone_env1.train_network(custom_network, 'data/env1/Comp3_movement_second/test15', is_pretrain=True)
    custom_network = [
                [
                    dict(type='retrieve', name='bbox_retrieve', tensors=['bbox']),
                    dict(type='dense', name='bbox_dense0', size=128, bias=True, activation='sigmoid'),
                    dict(type='dense', name='bbox_dense1', size=128, bias=True, activation='sigmoid'),
                    dict(type='register', name='bbox_register', tensor='bbox-embedding')
                ],
                [
                    dict(type='retrieve', name='velocity_retrieve', tensors=['velocity']),
                    dict(type='dense', name='velocity_dense0', size=128, bias=True, activation='sigmoid'),
                    dict(type='dense', name='velocity_dense1', size=128, bias=True, activation='sigmoid'),
                    dict(type='register', name='velocity_register', tensor='velocity-embedding')
                ],
                [
                    dict(type='retrieve', tensors=['bbox-embedding', 'velocity-embedding'], aggregation='concat'),
                    dict(type='dense', name='dense0', size=128, bias=True, activation='sigmoid'),
                ]
            ]
    drone_env1.train_network(custom_network, 'data/env1/Comp3_movement_second/test16', is_pretrain=True)

    # ########################################## Architecture 3 ##########################################
    custom_network = [
                [
                    dict(type='retrieve', name='bbox_retrieve', tensors=['bbox']),
                    dict(type='dense', name='bbox_dense0', size=128, bias=True, activation='tanh'),
                    dict(type='dense', name='bbox_dense1', size=256, bias=True, activation='tanh'),
                    dict(type='register', name='bbox_register', tensor='bbox-embedding')
                ],
                [
                    dict(type='retrieve', name='velocity_retrieve', tensors=['velocity']),
                    dict(type='dense', name='velocity_dense0', size=128, bias=True, activation='tanh'),
                    dict(type='dense', name='velocity_dense1', size=256, bias=True, activation='tanh'),
                    dict(type='register', name='velocity_register', tensor='velocity-embedding')
                ],
                [
                    dict(type='retrieve', tensors=['bbox-embedding', 'velocity-embedding'], aggregation='concat'),
                    dict(type='dense', name='dense0', size=256, bias=True, activation='tanh'),
                ]
            ]
    drone_env1.train_network(custom_network, 'data/env1/Comp3_movement_second/test17', is_pretrain=True)

    custom_network = [
                [
                    dict(type='retrieve', name='bbox_retrieve', tensors=['bbox']),
                    dict(type='dense', name='bbox_dense0', size=128, bias=True, activation='relu'),
                    dict(type='dense', name='bbox_dense1', size=256, bias=True, activation='relu'),
                    dict(type='register', name='bbox_register', tensor='bbox-embedding')
                ],
                [
                    dict(type='retrieve', name='velocity_retrieve', tensors=['velocity']),
                    dict(type='dense', name='velocity_dense0', size=128, bias=True, activation='relu'),
                    dict(type='dense', name='velocity_dense1', size=256, bias=True, activation='relu'),
                    dict(type='register', name='velocity_register', tensor='velocity-embedding')
                ],
                [
                    dict(type='retrieve', tensors=['bbox-embedding', 'velocity-embedding'], aggregation='concat'),
                    dict(type='dense', name='dense0', size=256, bias=True, activation='relu'),
                ]
            ]
    drone_env1.train_network(custom_network, 'data/env1/Comp3_movement_second/test18', is_pretrain=True)

    '''
    # Error
    custom_network = [
                [
                    dict(type='retrieve', name='bbox_retrieve', tensors=['bbox']),
                    dict(type='dense', name='bbox_dense0', size=128, bias=True, activation='crelu'),
                    dict(type='dense', name='bbox_dense1', size=256, bias=True, activation='crelu'),
                    dict(type='register', name='bbox_register', tensor='bbox-embedding')
                ],
                [
                    dict(type='retrieve', name='velocity_retrieve', tensors=['velocity']),
                    dict(type='dense', name='velocity_dense0', size=128, bias=True, activation='crelu'),
                    dict(type='dense', name='velocity_dense1', size=256, bias=True, activation='crelu'),
                    dict(type='register', name='velocity_register', tensor='velocity-embedding')
                ],
                [
                    dict(type='retrieve', tensors=['bbox-embedding', 'velocity-embedding'], aggregation='concat'),
                    dict(type='dense', name='dense0', size=256, bias=True, activation='crelu'),
                ]
            ]
    #drone_env1.train_network(custom_network, 'data/env1/Comp3_movement_second/test19', is_pretrain=True)
    '''

    custom_network = [
                [
                    dict(type='retrieve', name='bbox_retrieve', tensors=['bbox']),
                    dict(type='dense', name='bbox_dense0', size=128, bias=True, activation='elu'),
                    dict(type='dense', name='bbox_dense1', size=256, bias=True, activation='elu'),
                    dict(type='register', name='bbox_register', tensor='bbox-embedding')
                ],
                [
                    dict(type='retrieve', name='velocity_retrieve', tensors=['velocity']),
                    dict(type='dense', name='velocity_dense0', size=128, bias=True, activation='elu'),
                    dict(type='dense', name='velocity_dense1', size=256, bias=True, activation='elu'),
                    dict(type='register', name='velocity_register', tensor='velocity-embedding')
                ],
                [
                    dict(type='retrieve', tensors=['bbox-embedding', 'velocity-embedding'], aggregation='concat'),
                    dict(type='dense', name='dense0', size=256, bias=True, activation='elu'),
                ]
            ]
    drone_env1.train_network(custom_network, 'data/env1/Comp3_movement_second/test20', is_pretrain=True)

    custom_network = [
                [
                    dict(type='retrieve', name='bbox_retrieve', tensors=['bbox']),
                    dict(type='dense', name='bbox_dense0', size=128, bias=True, activation='leaky-relu'),
                    dict(type='dense', name='bbox_dense1', size=256, bias=True, activation='leaky-relu'),
                    dict(type='register', name='bbox_register', tensor='bbox-embedding')
                ],
                [
                    dict(type='retrieve', name='velocity_retrieve', tensors=['velocity']),
                    dict(type='dense', name='velocity_dense0', size=128, bias=True, activation='leaky-relu'),
                    dict(type='dense', name='velocity_dense1', size=256, bias=True, activation='leaky-relu'),
                    dict(type='register', name='velocity_register', tensor='velocity-embedding')
                ],
                [
                    dict(type='retrieve', tensors=['bbox-embedding', 'velocity-embedding'], aggregation='concat'),
                    dict(type='dense', name='dense0', size=256, bias=True, activation='leaky-relu'),
                ]
            ]
    drone_env1.train_network(custom_network, 'data/env1/Comp3_movement_second/test21', is_pretrain=True)

    custom_network = [
                [
                    dict(type='retrieve', name='bbox_retrieve', tensors=['bbox']),
                    dict(type='dense', name='bbox_dense0', size=128, bias=True, activation='none'),
                    dict(type='dense', name='bbox_dense1', size=256, bias=True, activation='none'),
                    dict(type='register', name='bbox_register', tensor='bbox-embedding')
                ],
                [
                    dict(type='retrieve', name='velocity_retrieve', tensors=['velocity']),
                    dict(type='dense', name='velocity_dense0', size=128, bias=True, activation='none'),
                    dict(type='dense', name='velocity_dense1', size=256, bias=True, activation='none'),
                    dict(type='register', name='velocity_register', tensor='velocity-embedding')
                ],
                [
                    dict(type='retrieve', tensors=['bbox-embedding', 'velocity-embedding'], aggregation='concat'),
                    dict(type='dense', name='dense0', size=256, bias=True, activation='none'),
                ]
            ]
    drone_env1.train_network(custom_network, 'data/env1/Comp3_movement_second/test22', is_pretrain=True)

    custom_network = [
                [
                    dict(type='retrieve', name='bbox_retrieve', tensors=['bbox']),
                    dict(type='dense', name='bbox_dense0', size=128, bias=True, activation='selu'),
                    dict(type='dense', name='bbox_dense1', size=256, bias=True, activation='selu'),
                    dict(type='register', name='bbox_register', tensor='bbox-embedding')
                ],
                [
                    dict(type='retrieve', name='velocity_retrieve', tensors=['velocity']),
                    dict(type='dense', name='velocity_dense0', size=128, bias=True, activation='selu'),
                    dict(type='dense', name='velocity_dense1', size=256, bias=True, activation='selu'),
                    dict(type='register', name='velocity_register', tensor='velocity-embedding')
                ],
                [
                    dict(type='retrieve', tensors=['bbox-embedding', 'velocity-embedding'], aggregation='concat'),
                    dict(type='dense', name='dense0', size=256, bias=True, activation='selu'),
                ]
            ]
    drone_env1.train_network(custom_network, 'data/env1/Comp3_movement_second/test23', is_pretrain=True)

    custom_network = [
                [
                    dict(type='retrieve', name='bbox_retrieve', tensors=['bbox']),
                    dict(type='dense', name='bbox_dense0', size=128, bias=True, activation='sigmoid'),
                    dict(type='dense', name='bbox_dense1', size=256, bias=True, activation='sigmoid'),
                    dict(type='register', name='bbox_register', tensor='bbox-embedding')
                ],
                [
                    dict(type='retrieve', name='velocity_retrieve', tensors=['velocity']),
                    dict(type='dense', name='velocity_dense0', size=128, bias=True, activation='sigmoid'),
                    dict(type='dense', name='velocity_dense1', size=256, bias=True, activation='sigmoid'),
                    dict(type='register', name='velocity_register', tensor='velocity-embedding')
                ],
                [
                    dict(type='retrieve', tensors=['bbox-embedding', 'velocity-embedding'], aggregation='concat'),
                    dict(type='dense', name='dense0', size=256, bias=True, activation='sigmoid'),
                ]
            ]
    drone_env1.train_network(custom_network, 'data/env1/Comp3_movement_second/test24', is_pretrain=True)

# ### Valid all results of Architecture 1, 2 and 3 env1 ###
if what_to_run == 33:
    drone_env1 = DroneEnv1()
    path_folders = 'data/env1/Comp3_movement_second'
    #folders_objective = ['test4']
    for main_folder in os.listdir(path_folders):
        if os.path.isdir(f'{path_folders}/{main_folder}'):
            #if main_folder in folders_objective:
            for checkpoint_folder in os.listdir(f'{path_folders}/{main_folder}'):
                if os.path.isdir(f'{path_folders}/{main_folder}/{checkpoint_folder}'):
                    print(f'{path_folders}/{main_folder}/{checkpoint_folder}')
                    shutil.rmtree('data/agent_checkpoints')
                    time.sleep(0.1)
                    shutil.copytree(f'{path_folders}/{main_folder}/{checkpoint_folder}', 'data/agent_checkpoints')
                    drone_env1.valid_network(path_csv=f'{path_folders}/{main_folder}/{checkpoint_folder}.csv')

# ### Valid tracker ###
if what_to_run == 4:
    drone_env1 = DroneEnv1()
    drone_env1.valid_network(epochs=20, path_csv='data/env1/Comp3_tracker/test1/tracker1.csv', use_tracker=True,
                             test_tracker=True,
                             dist_y_between_apple=0.075, dist_z_between_apple=0.08)
    drone_env1.valid_network(epochs=20, path_csv='data/env1/Comp3_tracker/test2/tracker2.csv', use_tracker=True,
                             test_tracker=True,
                             dist_y_between_apple=0.085, dist_z_between_apple=0.085)
    drone_env1.valid_network(epochs=20, path_csv='data/env1/Comp3_tracker/test3/tracker3.csv', use_tracker=True,
                             test_tracker=True,
                             dist_y_between_apple=0.090, dist_z_between_apple=0.095)

# ### Collect data to train suction ###
if what_to_run == 5:
    drone_env1 = DroneEnv1(is_save_data_suction_apple=True)
    shutil.rmtree('data/agent_checkpoints', ignore_errors=True)
    shutil.copytree(f'components/component3/agent', 'data/agent_checkpoints')
    drone_env1.valid_network(epochs=20000, path_csv=f'data/env1/Comp3_suction/historical.csv')

# ### Valid suction ###
if what_to_run == 55:
    drone_env1 = DroneEnv1(is_save_img_bbox_detection=False)
    drone_env1.valid_network(epochs=100, path_csv='data/env1/Comp3_suction/result_env1_suction.csv', test_suction=True, use_tracker=True)


# ### Valid env2 ###
if what_to_run == 6:
    # Only works if load network in the same path that as saved when trained
    shutil.rmtree('data/agent_checkpoints')
    time.sleep(0.1)
    shutil.copytree(f'components/component3/agent', 'data/agent_checkpoints')

    drone_env2 = DroneEnv2()
    drone_env2.valid_network(epochs=2, path_csv='data/env2/final_results/result.csv')
