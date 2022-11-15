#import warnings
#warnings.filterwarnings('ignore')
import time

import gym
import cv2
import numpy as np
import pandas as pd

from components.components1_and_4.components_1_and_4 import Component1_4
from components.component2.component_2 import Component2
from components.component3.component_3 import Component3

class DroneTensorforceEnv2:
    def __init__(self, is_save_img_bbox_detection=True):
        self.is_save_img_bbox_detection = is_save_img_bbox_detection

        # (w, h, c)
        self.input_shape_detector = (256, 144, 3)   # Environment return image shape
        # Epoch step limit
        self.nstep_limit = 9999999999
        # Collect apple step limit
        self.nstep_limit_collect = 100

        self.environment = gym.make(
            "airgym:airsim-drone-sample-v0",
            ip_address="127.0.0.1",
            step_length=0.5,
            nstep_limit=self.nstep_limit,
            image_shape=self.input_shape_detector,
            environment=2,
            threshold_spawn_apple=40
        )

        self.component2 = Component2()
        self.component3 = Component3()
        self.component1_4 = Component1_4(self.environment)
        # Set to not pick the apple when touching in it
        self.environment.set_auto_grab(False)

    def run(self, epochs=400, path_csv='historical.csv'):
        """

        Args:
            epochs (int): Number of epochs (trees) to execute
            path_csv (str): Path to save the historical
        """
        # Create files to storage historical
        path_collected_csv = f'{path_csv[:path_csv.rfind(".")]}_collected.csv'
        columns_csv = ['episode', 'amount_collected', 'steps', 'steps_try_grab_apple',
                       'time', 'mean_time', 'mean_fps', 'time_deposit', 'total_time', 'last_code_grab_overlap',
                       'last_code_arm_collision', 'last_drone_collision', 'last_action', 'last_state']
        pd.DataFrame(columns=columns_csv).to_csv(path_collected_csv, index=False, sep=';')

        path_episode_csv = f'{path_csv[:path_csv.rfind(".")]}_episode.csv'
        columns_csv_episode = ['episode', 'NR_collected', 'NG_collected', 'OR_collected', 'OG_collected',
                               'NR_available', 'NG_available', 'OR_available', 'OG_available']
        pd.DataFrame(columns=columns_csv_episode).to_csv(path_episode_csv, index=False, sep=';')

        path_change_position_csv = f'{path_csv[:path_csv.rfind(".")]}_change_position.csv'
        columns_csv_change_position = ['episode', 'steps', 'time_before_movement', 'time_movement', 'direction', 'last_code_grab_overlap',
                                       'last_code_arm_collision', 'last_drone_collision']
        pd.DataFrame(columns=columns_csv_change_position).to_csv(path_change_position_csv, index=False, sep=';')

        # Run for n episodes
        for eps in range(epochs):
            try:
                # Record episode experience
                episode_states = []
                episode_actions = []
                episode_obs = []
                # Apples collected in the episode
                episode_apples_collected = [0, 0, 0, 0]
                episode_total_apples = [0, 0, 0, 0]
                # 0 == Default; 1 == Collected apple; 2 == Changed position (new position); 3 == Collision (fall back)
                status = 0
                deposit_apple_time = 0


                # Count all attempts to collect apple
                count_ends = 0

                _, state = self.environment.reset()
                # Total NR, NG, OR and OG available
                episode_total_apples[0] = int(str(state['code_amount_apple'])[1:3])
                episode_total_apples[1] = int(str(state['code_amount_apple'])[3:5])
                episode_total_apples[2] = int(str(state['code_amount_apple'])[5:7])
                episode_total_apples[3] = int(str(state['code_amount_apple'])[7:9])

                is_continue = True
                self.component1_4.start_epoch(state['drone_position'].x_val)
                last_is_moving_up = self.component1_4.is_moving_up
                direction = 'up'

                # while all apples in the tree were not harvested
                while is_continue:
                    all_img_bbox_detection = []
                    self.environment.reset_status()
                    is_suck_apple = False
                    count_step_try_grab_apple = 0

                    # To initialize some drone's data
                    obs, reward, done, state, _ = self.environment.step(action=0, is_suck_apple=is_suck_apple)

                    self.component3.start_agent()

                    done = False
                    step = 1

                    tic = time.time()

                    step_collect = -1
                    while not done:
                        step += 1
                        step_collect += 1
                        # To many steps without collect apple (system problem or simulator bugged)
                        if step_collect >= self.nstep_limit_collect:
                            break

                        classes, confidences, boxes_detector = self.component2.detect(obs)
                        if self.is_save_img_bbox_detection:
                            all_img_bbox_detection.append([obs, classes, confidences, boxes_detector, False])

                        idx_closest_apple = 0
                        all_distances = []
                        if boxes_detector:
                            idx_closest_apple, all_distances = self.component2.track(np.array(boxes_detector)[:, :2],
                                                                                     classes)

                        drone_velocity = (float(state['drone_velocity'].x_val), float(state['drone_velocity'].y_val),
                                           float(state['drone_velocity'].z_val))

                        # Not detected apples to be harvest
                        if not all_distances:
                            is_continue = self.component1_4.update_drone_next_position()
                            episode_obs.append(obs)
                            break

                        input_agent = ({'bbox': boxes_detector[idx_closest_apple],
                                        'velocity': drone_velocity})
                        actions = self.component3.act_agent(input_agent)

                        if not is_suck_apple:
                            is_suck_apple = self.component3.suction_predict(list(input_agent['bbox']), list(input_agent['velocity']))
                        if is_suck_apple:
                            count_step_try_grab_apple += 1
                        if self.is_save_img_bbox_detection:
                            all_img_bbox_detection[-1][4] = is_suck_apple

                        obs, reward, done, state, actions = self.environment.step(action=actions,
                                                                                  is_suck_apple=is_suck_apple)

                        # If detector detected an apple or done, save data
                        if boxes_detector != [[0., 0., 0., 0.]] or done:
                            episode_states.append(input_agent)
                            episode_actions.append(actions)
                            episode_obs.append(obs)

                    eps_time = time.time() - tic

                    if is_continue:
                        tic = time.time()
                        if state['code_grab_overlap'] >= 21000:
                            status = 1
                            # Add apple collected
                            episode_apples_collected[int(str(state['code_grab_overlap'])[1]) - 1] += 1

                            self.component1_4.apple_collected(state['drone_position'].z_val)

                        else:
                            # Collision without grab apple
                            if state['code_arm_collision'] > 10000 or state['drone_collision'] is True:
                                self.component1_4.collision()
                                status = 3
                            # Moved some steps and stops to detect a NR
                            elif state['code_arm_collision'] == 10000 and step_collect > 5:
                                self.component1_4.collision(state['drone_position'].x_val)
                                status = 2
                            else:
                                # Not detected apple (change position)
                                status = 2

                            self.component1_4.change_position()

                        # Time to deposit or change position
                        deposit_apple_time = time.time() - tic

                    for idx, data in enumerate(all_img_bbox_detection):
                        img = self.component2.yolov7.draw_box(data[0], data[1], data[2], data[3], box_normalized=True,
                                                   box_scaled=True, write_label=False)

                        cv2.imwrite(f'data/env2/Comp2_bbox_detections/{eps}-{sum(episode_apples_collected)}-{count_ends}-{idx}-{data[4]}.png', img)

                    # Print all movement of the epoch
                    # for idx, ac in enumerate(episode_actions):
                    #    print(idx, episode_actions[idx], episode_states[idx])

                    if status in [1, 3]:
                        print('*************COLLECTED*************')
                        print(f'Episode: {eps}')
                        print(f'Amount apple collected: {sum(episode_apples_collected)}')
                        print(f'Steps: {step}')
                        print(f'Steps try grab: {count_step_try_grab_apple}')
                        print(f'Time: {eps_time}')
                        print(f'Mean time: {eps_time / step}')
                        print(f'Mean FPS: {1 / (eps_time / step)}')
                        print(f'Time deposit: {deposit_apple_time}')
                        print(f'Total time: {eps_time + deposit_apple_time}')

                        print(f'Last code_grab_overlap: {state["code_grab_overlap"]}')
                        print(f'Last code_arm_collision: {state["code_arm_collision"]}')
                        print(f'Last drone_collision: {state["drone_collision"]}')
                        print('***********************************')

                        if len(episode_states) > 0:
                            data_to_csv = [[eps, sum(episode_apples_collected), step, count_step_try_grab_apple, eps_time, eps_time / step,
                                            1 / (eps_time / step), deposit_apple_time, eps_time + deposit_apple_time, state["code_grab_overlap"],
                                            state["code_arm_collision"], state["drone_collision"],
                                            episode_actions[-1],
                                            episode_states[-1]]]
                            pd.DataFrame(data_to_csv).to_csv(path_collected_csv, sep=';', index=False, header=False, mode='a')
                    elif status == 2:
                        data_to_csv = [[eps, step, eps_time, deposit_apple_time, direction, state["code_grab_overlap"],
                                        state["code_arm_collision"], state["drone_collision"]]]
                        if last_is_moving_up != self.component1_4.is_moving_up:
                            data_to_csv[0][4] = 'left'
                            if direction == 'up':
                                direction = 'down'
                            else:
                                direction = 'up'
                        last_is_moving_up = self.component1_4.is_moving_up
                        pd.DataFrame(data_to_csv).to_csv(path_change_position_csv, sep=';', index=False, header=False,
                                                         mode='a')

                        print('**********CHANGED POSITION*********')
                        print(f'Episode: {eps}')
                        print(f'Steps: {step}')
                        print(f'Time before movement: {eps_time}')
                        print(f'Time movement: {deposit_apple_time}')
                        print(f'Direction: {data_to_csv[0][4]}')
                        print(f'Last code_grab_overlap: {state["code_grab_overlap"]}')
                        print(f'Last code_arm_collision: {state["code_arm_collision"]}')
                        print(f'Last drone_collision: {state["drone_collision"]}')
                        print('***********************************')

                        # Save last obs before change position
                        if len(episode_obs) > 0:
                            cv2.imwrite(f'data/env2/Comp1_change_position/{eps}-{self.component1_4.count_time_left}-{self.component1_4.count_time_up_down}.png',
                                        episode_obs[-1])
                            if len(all_img_bbox_detection) > 0:
                                img = self.component2.yolov7.draw_box(all_img_bbox_detection[-1][0], all_img_bbox_detection[-1][1],
                                                           all_img_bbox_detection[-1][2], all_img_bbox_detection[-1][3],
                                                           box_normalized=True, box_scaled=True,
                                                           write_label=False)
                                cv2.imwrite(f'data/env2/Comp1_change_position/{eps}-{self.component1_4.count_time_left}-{self.component1_4.count_time_up_down}-detection.png',
                                            img)

                    status = 0
                    count_ends += 1

                print('############END EPISODE############')
                print(f'Episode: {eps}')
                # NR, NG, OR, OG
                print(f'Apples collected: {episode_apples_collected}')
                print(f'Apples were available: {episode_total_apples}')
                print('###################################')

                data_to_csv = [[eps, episode_apples_collected[0], episode_apples_collected[1],
                                episode_apples_collected[2], episode_apples_collected[3], episode_total_apples[0],
                                episode_total_apples[1], episode_total_apples[2], episode_total_apples[3]]]
                pd.DataFrame(data_to_csv).to_csv(path_episode_csv, sep=';', index=False, header=False, mode='a')

                #input('Press ENTER to continue to the next episode!')

            except Exception as e:
                #print(e)
                raise e
