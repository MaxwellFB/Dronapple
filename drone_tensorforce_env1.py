"""
Used to train and valid the models in the environment 1
"""

import os
import shutil
import time
from tqdm import tqdm

import gym
import cv2
import pandas as pd
import numpy as np

from components.component2.yolov7.yolov7 import Yolov7Detection
from components.component2.tracker import Tracker
from components.component3.suction.suction import Suction


class DroneTensorforceEnv1:
    def __init__(self, path_save_checkpoint='data/agent_checkpoints', is_save_img_bbox_detection=False,
                 is_save_data_suction_apple=False):
        self.is_save_img_bbox_detection = is_save_img_bbox_detection
        self.is_save_data_suction_apple = is_save_data_suction_apple
        self.yolov7 = Yolov7Detection(path_model='models/yolov7-tiny-416-32-env1.pt', img_size=416, stride=32,
                                      conf_threshold=0.25, iou_threshold=0.45)
        self.suction = Suction(path_model='model/suction_network.pth', path_scaler='model/suction_std_scaler.bin')
        self.tracker = Tracker(objective_classes=[0])

        # (w, h, c)
        self.input_shape_detector = (256, 144, 3)  # Environment return image shape
        self.batch_size = 64
        # PS: First step do not collect data to experience
        self.nstep_limit = 100

        self.path_save_checkpoint = path_save_checkpoint

        # Limit of step without detect an apple until end the episode
        self.limit_not_detected = 3

        # Can be changed by the def set_exploration
        self.type_exploration = 801
        self.current_exploration = 0.35
        self.min_exploration = 0.35
        self.decay_exploration = 0.0001
        self.ignore_exploration_each_n_epoch = 5

        self.environment = gym.make(
            "airgym:airsim-drone-sample-v0",
            ip_address="127.0.0.1",
            step_length=0.5,
            nstep_limit=self.nstep_limit,
            image_shape=self.input_shape_detector,
            environment=1,
            threshold_spawn_apple=40
        )

        # Apple bounding box and drone velocity (inputs)
        self.my_states = {
            'bbox':
                {
                    'shape': 4,
                    'type': 'float',
                    'min_value': 0.,
                    'max_value': 1.
                },
            'velocity':
                {
                    'shape': 3,
                    'type': 'float',
                    'min_value': -1.,
                    'max_value': 1.
                }
        }

        # Velocity to move the drone coordinates X, Y and Z (output)
        self.my_actions = {
            'shape': 3,
            'type': 'float',
            'min_value': -1.,
            'max_value': 1.
        }

        self.agent = None

    def set_exploration(self, type_exploration=801, current_exploration=0.35, min_exploration=0.35,
                        decay_exploration=0.0001, ignore_exploration_each_n_epoch=5):
        """
        Configure the exploration when training agent

        Args:
            type_exploration (int): 800 == smart exploration; 801 == noise exploration
            current_exploration (float): Percentage of current chance to do an exploration
            min_exploration (float): Minimum percentage of chance to do an exploration
            decay_exploration (float): How many to decay the exploration each time the exploration happens
            ignore_exploration_each_n_epoch (int): Each N epoch, ignore during all epoch the chance to do exploration
        """
        self.type_exploration = type_exploration
        self.current_exploration = current_exploration
        self.min_exploration = min_exploration
        self.decay_exploration = decay_exploration
        self.ignore_exploration_each_n_epoch = ignore_exploration_each_n_epoch

    def create_agent(self, agent='ppo', network='auto', learning_rate=0.0001):
        """
        Create a new agent

        Args:
            agent (str): Agent type (ppo or ddpg)
            network (str or dict): Agent network
            learning_rate (float): Agent learning rate
        """
        # Gym does not works if Tensorforce is loaded before "gym.make"
        from tensorforce.agents import Agent

        if agent == 'ppo':
            self.agent = Agent.create(
                agent='ppo', states=self.my_states, actions=self.my_actions, memory='minimum',
                batch_size=self.batch_size, network=network, max_episode_timesteps=self.nstep_limit,
                learning_rate=learning_rate, saver=dict(directory=self.path_save_checkpoint, frequency=1,
                                                        max_checkpoints=3)
            )
        elif agent == 'ddpg':
            self.agent = Agent.create(
                agent='ddpg', states=self.my_states, actions=self.my_actions, memory=5000, batch_size=self.batch_size,
                network=network, max_episode_timesteps=self.nstep_limit, learning_rate=learning_rate,
                l2_regularization=0.1, entropy_regularization=0.1, saver=dict(directory=self.path_save_checkpoint,
                                                                              frequency=1, max_checkpoints=3)
            )
        else:
            raise ValueError('Invalid agent!')

        print(self.agent.get_specification())
        print('******************')
        print(self.agent.get_architecture())

    def load_agent(self):
        """
        Load agent from self.path_save_checkpoint
        """
        # Gym does not works if Tensorforce is loaded before "gym.make"
        from tensorforce.agents import Agent
        # Only works if load network in the same path that as saved when trained
        self.agent = Agent.load(directory=self.path_save_checkpoint)
        print(self.agent.get_specification())
        print('******************')
        print(self.agent.get_architecture())

    def pretrain(self, path_data, max_files_pretrain=100):
        """
         Train using history
        """
        count = 0
        for filename in tqdm(os.listdir(path_data)):
            if not filename.endswith('.npz'):
                continue
            if count >= max_files_pretrain:
                break
            count += 1

            file = np.load(f'{path_data}/{filename}', allow_pickle=True)
            self.agent.experience(
                states=list(file['states']), internals=list(file['internals']), actions=list(file['actions']),
                terminal=list(file['done']), reward=list(file['reward'])
            )
            self.agent.update()

    def run(self, epochs=400, do_update=True, manual_system=False, save_best=True, path_recorder='data/env1/recorder',
            path_backup_checkpoint='data/backup', path_csv='data/backup/historical.csv', test_suction=False,
            use_tracker=False, test_tracker=False, dist_y_between_apple=0.075, dist_z_between_apple=0.08):
        """
        Args:
            epochs (int): Number of epochs to execute
            do_update (bool): Whether to update the network each epoch
            manual_system (bool): Whether to run the manual system
            save_best (bool): Whether to save the model every best reward and time
            path_recorder (str): Path to save state of each movement
            path_backup_checkpoint (str): Path to save the checkpoints
            path_csv (str): Path to save the historical
            test_suction (bool): Whether to predict when to grab the apple
            use_tracker (bool): Whether to use the tracker
            test_tracker (bool): Whether to to test the tracker (environment with 9 apples) (use_tracker==True)
            dist_y_between_apple (float): Distance Y between apples (test_tracker==True)
            dist_z_between_apple (float): Distance Z between apples (test_tracker==True)
        """
        # Whether to pick the apple automatically when touching in it
        if test_suction:
            self.environment.set_auto_grab(False)
        else:
            self.environment.set_auto_grab(True)

        # Always red apple (NR)
        self.environment.disable_percentage_apple()

        # The model for env1 were trained with only one apple, to test tracker needs to have more them one apple
        if test_tracker:
            self.yolov7 = Yolov7Detection(path_model='models/yolov7-tiny-416-32-env2.pt', img_size=416,
                                          stride=32, conf_threshold=0.25, iou_threshold=0.45)
            # In this case the class is not important
            self.tracker = Tracker(objective_classes=[0, 1, 2, 3])

        all_rewards = []
        best_reward = -9999999
        best_time = 9999999
        best_eps_reward = -1
        best_eps_time = -1

        count_backup_reward = 0
        count_backup_time = 0

        # Count all images to continue the ID (filename)
        if self.is_save_data_suction_apple:
            count_save_suction_apple = (len(os.listdir('data/env1/Comp3_suction/suck_apple/image')) +
                                        len(os.listdir('data/env1/Comp3_suction/fail_suck_apple/image')) +
                                        len(os.listdir('data/env1/Comp3_suction/not_suck_apple/image')))

        if self.is_save_img_bbox_detection:
            count_save_bbox_detection = len(os.listdir('data/env1/Comp2_bbox_detections'))

        # Create files to storage historical
        columns_csv = ['episode', 'steps', 'steps_try_grab_apple', 'initial_position_apple', 'current_exploration',
                       'time', 'mean_time', 'mean_fps', 'reward', 'last_reward', 'last_code_grab_overlap',
                       'last_code_arm_collision', 'last_drone_collision', 'last_action', 'last_state']
        pd.DataFrame(columns=columns_csv).to_csv(path_csv, index=False, sep=';')

        # Run for n episodes
        for eps in range(epochs):
            try:
                # Record episode experience
                episode_states = []
                episode_internals = []
                episode_actions = []
                episode_done = []
                episode_reward = []
                episode_obs = []

                self.environment.reset(test_tracker=test_tracker, dist_y_between_apple=dist_y_between_apple,
                                       dist_z_between_apple=dist_z_between_apple)
                is_suck_apple = False
                count_step_try_grab_apple = 0

                # To initialize some drone's data
                obs, reward, done, state, _ = self.environment.step(action=0, is_suck_apple=is_suck_apple)

                initial_position_apple = (state['apple_position'].x_val, state['apple_position'].y_val,
                                          state['apple_position'].z_val)

                internals = self.agent.initial_internals()
                done = False
                sum_reward = 0
                step = 1

                count_detector_not_detected = 0
                tic = time.time()
                all_img_bbox_detection = []

                while not done:
                    step += 1

                    classes, confidences, boxes_detector = self.yolov7.detect(obs, box_normalize=True, box_scale=True)
                    if self.is_save_img_bbox_detection:
                        all_img_bbox_detection.append(
                            self.yolov7.draw_box(obs, classes, confidences, boxes_detector, box_normalized=True,
                                                 box_scaled=True))

                    idx_closest_apple = 0
                    if use_tracker:
                        if boxes_detector:
                            idx_closest_apple, d = self.tracker.get_distance_object(main_position=(0.45, 0.55),
                                                                                    object_positions=np.array(
                                                                                        boxes_detector)[:, :2],
                                                                                    classes=classes)

                    drone_velocity = (float(state['drone_velocity'].x_val), float(state['drone_velocity'].y_val),
                                      float(state['drone_velocity'].z_val))

                    # Movement the drone using logical conditions and environment information
                    if manual_system:
                        save_best = False

                        if not boxes_detector:
                            actions = (0.1, 0.1, 0.1)
                            boxes_detector = [[0., 0., 0., 0.]]
                            count_detector_not_detected += 1
                        else:
                            count_detector_not_detected = 0
                            # Default speed
                            x, y, z = 0.102, 0.0, -0.17

                            # ### Foward
                            if state['drone_position'].x_val + 0.2 < state['apple_position'].x_val:
                                x = 0.2
                            if state['drone_position'].x_val + 0.45 < state['apple_position'].x_val:
                                x = 0.4

                            # ### Left
                            if boxes_detector[0][0] < 0.455:
                                # Too close
                                y = -0.015

                                # Close
                                if boxes_detector[0][0] < 0.44:
                                    y = -0.05

                                # Far - More velocity
                                if boxes_detector[0][0] < 0.34 and drone_velocity[1] > -0.3:
                                    # x = 0.0
                                    y = -0.4
                                if boxes_detector[0][0] < 0.34 and drone_velocity[1] > -0.2:
                                    # x = 0.0
                                    y = -0.6
                                if boxes_detector[0][0] < 0.28 and drone_velocity[1] > -0.2:
                                    # x = 0.0
                                    y = -0.9

                                # More or less close - Less velocity
                                if boxes_detector[0][0] < 0.42 and boxes_detector[0][0] > 0.34:
                                    # x = 0.0
                                    y = -0.15
                                if boxes_detector[0][0] < 0.34 and boxes_detector[0][0] > 0.30 and \
                                        drone_velocity[1] < -0.1:
                                    # x = 0.0
                                    y = -0.3

                                # Reduce speed
                                if boxes_detector[0][0] < 0.45 and boxes_detector[0][0] > 0.42 and \
                                        drone_velocity[1] < -0.05:  # 0 e 16
                                    y = 0.1
                                if boxes_detector[0][0] < 0.44 and boxes_detector[0][0] > 0.28 \
                                        and drone_velocity[1] < -0.15:
                                    y = 0.15
                                if boxes_detector[0][0] < 0.44 and boxes_detector[0][0] > 0.34 \
                                        and drone_velocity[1] < -0.15:
                                    y = 0.3
                                if boxes_detector[0][0] < 0.44 and boxes_detector[0][0] > 0.34 \
                                        and drone_velocity[1] < -0.20:
                                    y = 0.6

                            # ### Right
                            elif boxes_detector[0][0] > 0.475:  # 45
                                # Too close
                                y = 0.015

                                # Far - More velocity
                                if boxes_detector[0][0] > 0.57 and drone_velocity[1] < 0.3:  # 10
                                    # x = 0.0
                                    y = 0.4
                                if boxes_detector[0][0] > 0.57 and drone_velocity[1] < 0.2:  # 10
                                    # x = 0.0
                                    y = 0.6
                                if boxes_detector[0][0] > 0.63 and drone_velocity[1] < 0.2:  # 16
                                    # x = 0.0
                                    y = 0.9

                                # More or less close - Less velocity
                                if boxes_detector[0][0] > 0.50 and boxes_detector[0][0] < 0.57:  # 2 e 10
                                    # x = 0.0
                                    y = 0.15
                                if boxes_detector[0][0] > 0.57 and boxes_detector[0][0] < 0.61 and \
                                        drone_velocity[1] > 0.1:  # 10 e 14
                                    # x = 0.0
                                    y = 0.3

                                # Reduce speed
                                if boxes_detector[0][0] > 0.47 and boxes_detector[0][0] < 0.52 and \
                                        drone_velocity[1] > 0.04:  # 0 e 16
                                    y = -0.1
                                if boxes_detector[0][0] > 0.47 and boxes_detector[0][0] < 0.64 and \
                                        drone_velocity[1] > 0.15:  # 0 e 16
                                    y = -0.15
                                if boxes_detector[0][0] > 0.47 and boxes_detector[0][0] < 0.58 and \
                                        drone_velocity[1] > 0.15:  # 0 e 10
                                    y = -0.3
                                if boxes_detector[0][0] > 0.47 and boxes_detector[0][0] < 0.58 and \
                                        drone_velocity[1] > 0.20:  # 0 e 10
                                    y = -0.6

                            # ### Between left and right. Try to change Y velocity to 0
                            else:
                                if drone_velocity[1] > 0.01:
                                    y = -0.01
                                elif drone_velocity[1] < -0.01:
                                    y = 0.01

                                if drone_velocity[1] > 0.03:
                                    y = -0.02
                                elif drone_velocity[1] < -0.03:
                                    y = 0.02

                                if drone_velocity[1] > 0.05:
                                    y = -0.028
                                elif drone_velocity[1] < -0.05:
                                    y = 0.028

                            # ### Up
                            if boxes_detector[0][1] < 0.56:
                                # Close
                                z = -0.3  # -0.05

                                if boxes_detector[0][1] < 0.46 and drone_velocity[2] > -0.4:
                                    # x = 0.0
                                    z = -0.4
                                if boxes_detector[0][1] < 0.46 and drone_velocity[2] > -0.3:
                                    # x = 0.0
                                    z = -0.6
                                if boxes_detector[0][1] < 0.40 and drone_velocity[2] > -0.3:
                                    # x = 0.0
                                    z = -0.9

                            # ### Down
                            elif boxes_detector[0][1] > 0.62:  # 0.62:
                                z = 0.1
                                # if boxes_detector[0][1] > 0.72:
                                # x = 0.0
                                #    z = 0.3

                            # ### Between up and down. Try to change Z velocity to 0
                            else:
                                if boxes_detector[0][1] > 0.58:
                                    z = -0.15
                                if boxes_detector[0][1] > 0.60:
                                    z = -0.11

                            actions = (x, y, z)

                        input_agent = ({'bbox': boxes_detector[idx_closest_apple],
                                        'velocity': drone_velocity})
                        # Just to get the internals
                        _, new_internals = self.agent.act(states=input_agent, internals=internals, independent=True)
                    else:
                        if not boxes_detector:
                            boxes_detector = [[0., 0., 0., 0.]]
                            count_detector_not_detected += 1
                        else:
                            count_detector_not_detected = 0

                        input_agent = ({'bbox': boxes_detector[idx_closest_apple],
                                        'velocity': drone_velocity})
                        actions, new_internals = self.agent.act(states=input_agent, internals=internals,
                                                                independent=True)

                        if test_suction:
                            if not is_suck_apple:
                                is_suck_apple = self.suction.predict(tuple(list(input_agent['bbox']) +
                                                                           list(input_agent['velocity'])))
                            if is_suck_apple:
                                count_step_try_grab_apple += 1

                        # Convert to tuple and Python float
                        actions = (actions[0].item(), actions[1].item(), actions[2].item())

                        # Ignore exploration each n epochs
                        if eps % self.ignore_exploration_each_n_epoch != 0:
                            if np.random.rand(1)[0] <= self.current_exploration:
                                # Smart exploration
                                if self.type_exploration == 800:
                                    actions = self.type_exploration
                                # Noise exploration
                                else:
                                    actions = [actions[0] + np.random.uniform(-0.05, 0.05),
                                               actions[1] + np.random.uniform(-0.05, 0.05),
                                               actions[2] + np.random.uniform(-0.05, 0.05)]
                                    # Avoid values bigger then 1. and smaller then -1.
                                    if actions[0] > 1.:
                                        actions[0] = 1.
                                    elif actions[0] < -1.:
                                        actions[0] = -1.
                                    if actions[1] > 1.:
                                        actions[1] = 1.
                                    elif actions[1] < -1.:
                                        actions[1] = -1.
                                    if actions[2] > 1.:
                                        actions[2] = 1.
                                    elif actions[2] < -1.:
                                        actions[2] = -1.
                                    actions = tuple(actions)

                                # Exploration decay
                                if self.current_exploration > self.min_exploration:
                                    self.current_exploration -= self.decay_exploration

                    obs, reward, done, state, actions = self.environment.step(action=actions,
                                                                              is_suck_apple=is_suck_apple)
                    sum_reward += reward

                    # If detector detected an apple
                    if boxes_detector != [[0., 0., 0., 0.]] or done:
                        episode_states.append(input_agent)
                        episode_internals.append(internals)
                        internals = new_internals.copy()
                        episode_actions.append(actions)
                        episode_done.append(done)
                        episode_reward.append(reward)
                        episode_obs.append(obs)

                    # If detector did not detect an apple, change the last step that detected
                    elif count_detector_not_detected >= self.limit_not_detected:
                        done = True
                        if len(episode_done) > 0:
                            episode_done[-1] = done
                            episode_reward[-1] = self.environment.calculate_reward(is_failed=True)
                            sum_reward = sum_reward - reward + episode_reward[-1]
                            reward = episode_reward[-1]

                eps_time = time.time() - tic
                for idx in range(len(all_img_bbox_detection)):
                    count_save_bbox_detection += 1
                    cv2.imwrite(f'data/env1/Comp2_bbox_detections/{count_save_bbox_detection}-{idx}.png',
                                all_img_bbox_detection[idx])

                if self.is_save_data_suction_apple:
                    for idx in range(len(episode_obs) - 1):
                        count_save_suction_apple += 1
                        # Save as grab before really grab the apple
                        if not episode_done[idx + 1]:
                            cv2.imwrite(f'data/env1/Comp3_suction/not_suck_apple/image/{count_save_suction_apple}.png',
                                        episode_obs[idx])
                            pd.DataFrame(
                                [list(episode_states[idx]['bbox']) + list(episode_states[idx]['velocity'])]).to_csv(
                                f'data/env1/Comp3_suction/not_suck_apple/bbox/{count_save_suction_apple}.csv', sep=';',
                                header=False,
                                index=False
                            )
                        else:
                            if episode_reward[idx + 1] > 600:
                                cv2.imwrite(f'data/env1/Comp3_suction/suck_apple/image/{count_save_suction_apple}.png',
                                            episode_obs[idx])
                                pd.DataFrame(
                                    [list(episode_states[idx]['bbox']) + list(episode_states[idx]['velocity'])]).to_csv(
                                    f'data/env1/Comp3_suction/suck_apple/bbox/{count_save_suction_apple}.csv', sep=';',
                                    header=False,
                                    index=False
                                )
                            elif episode_reward[idx + 1] < 0:
                                cv2.imwrite(
                                    f'data/env1/Comp3_suction/fail_suck_apple/image/{count_save_suction_apple}.png',
                                    episode_obs[idx])
                                pd.DataFrame(
                                    [list(episode_states[idx]['bbox']) + list(episode_states[idx]['velocity'])]).to_csv(
                                    f'data/env1/Comp3_suction/fail_suck_apple/bbox/{count_save_suction_apple}.csv',
                                    sep=';', header=False,
                                    index=False
                                )

                if save_best:
                    if sum_reward > best_reward and sum_reward > 600:
                        best_reward = sum_reward
                        best_eps_reward = eps
                        count_backup_reward += 1

                        if os.path.exists(f'{path_backup_checkpoint}/reward_{count_backup_reward}'):
                            shutil.rmtree(f'{path_backup_checkpoint}/reward_{count_backup_reward}')
                            time.sleep(0.1)
                        shutil.copytree(self.path_save_checkpoint,
                                        f'{path_backup_checkpoint}/reward_{count_backup_reward}')
                        print('NEW BEST REWARD')
                    if eps_time < best_time and sum_reward > 600:
                        best_time = eps_time
                        best_eps_time = eps
                        count_backup_time += 1

                        if os.path.exists(f'{path_backup_checkpoint}/time_{count_backup_time}'):
                            shutil.rmtree(f'{path_backup_checkpoint}/time_{count_backup_time}')
                            time.sleep(0.1)
                        shutil.copytree(self.path_save_checkpoint, f'{path_backup_checkpoint}/time_{count_backup_time}')
                        print('NEW BEST TIME')

                if len(episode_actions) > 1:
                    data_to_csv = [
                        [eps, step, count_step_try_grab_apple, initial_position_apple, self.current_exploration, eps_time,
                         eps_time / step,
                         1 / (eps_time / step), sum_reward, reward, state["code_grab_overlap"],
                         state["code_arm_collision"], state["drone_collision"], episode_actions[-1],
                         episode_states[-1]]]
                    pd.DataFrame(data_to_csv).to_csv(path_csv, sep=';', index=False,
                                                     header=False, mode='a')

                # Print all movement of the epoch
                # for idx, ac in enumerate(episode_actions):
                #    print(idx, episode_actions[idx], episode_states[idx])

                print('**********************END EPISODE***********************************')
                print(f'Episode: {eps}')
                print(f'Steps: {step}')
                print(f'Steps try grab: {count_step_try_grab_apple}')
                print(f'Initial position apple: {initial_position_apple}')
                print(f'Exploration: {self.current_exploration}')
                print(f'Time: {eps_time}')
                print(f'Mean time: {eps_time / step}')
                print(f'Mean FPS: {1 / (eps_time / step)}')

                all_rewards.append(sum_reward)
                print(f'Reward: {sum_reward}')
                print(f'Last reward: {reward}')
                print(f'Last code_grab_overlap: {state["code_grab_overlap"]}')
                print(f'Last code_arm_collision: {state["code_arm_collision"]}')
                print(f'Last drone_collision: {state["drone_collision"]}')
                # print(f'Top episode: {np.argmax(all_rewards)} - reward {max(all_rewards)}')
                print(f'Top episode reward: {best_eps_reward} - reward {best_reward}')
                print(f'Top episode time: {best_eps_time} - time {best_time}')
                print('********************************************************************')

                # Write recorded episode trace to npz file
                if len(episode_done) > 0:
                    np.savez_compressed(
                        file=os.path.join(path_recorder, 'trace-{:09d}.npz'.format(eps)),
                        states=np.stack(episode_states, axis=0),
                        actions=np.stack(episode_actions, axis=0),
                        done=np.stack(episode_done, axis=0),
                        reward=np.stack(episode_reward, axis=0),
                        internals=np.stack(episode_internals, axis=0)
                    )

                    if do_update:
                        # Feed recorded experience to agent
                        self.agent.experience(
                            states=episode_states, internals=episode_internals, actions=episode_actions,
                            terminal=episode_done, reward=episode_reward
                        )

                        # Perform update
                        self.agent.update()
            except Exception as e:
                # print(e)
                raise e
