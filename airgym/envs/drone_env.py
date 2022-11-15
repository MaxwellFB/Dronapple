import time
import random

import airsim
import numpy as np
from PIL import Image
from gym import spaces

from airgym.envs.airsim_env import AirSimEnv


class AirSimDroneEnv(AirSimEnv):
    def __init__(self, ip_address, step_length, nstep_limit, image_shape, environment, threshold_spawn_apple=40):
        """

        Args:
            ip_address (str): Ip address to connect with Unreal (Airsim)
            step_length (float): Drone's movement velocity, used if the action is an int
            nstep_limit (int): Limit of steps per epoch
            image_shape (tuple[int,int,int]): Shape of drone's camera image (width, height, channel)
            environment (int): Number of the environment in Unreal
            threshold_spawn_apple (int): Threshold between 0 and 100 to spawn apple, where 100 always spawn and
             0 never spawn


        """
        print('__init__')
        super().__init__(image_shape)
        self.step_length = step_length
        self.image_shape = image_shape
        self.environment = environment
        self.threshold_spawn_apple = threshold_spawn_apple
        self.nstep = 0
        self.nstep_limit = nstep_limit
        self.prev_dist = 0
        self.code_amount_apple = 0

        self.is_overlapping_normal_apple = 0
        self.velocity_grab_apple = 0.0
        self.diff_distance = 0

        self.state = {
            'drone_position': np.zeros(3),
            'prev_drone_position': np.zeros(3),
            'drone_velocity': np.zeros(3),
            'drone_collision': False,
            'code_arm_collision': 10000,
            'code_grab_overlap': 10000,
            'apple_position': np.zeros(3)
        }

        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.drone.reset()
        self.action_space = spaces.Discrete(8)

        self.image_request = [airsim.ImageRequest(
            0, airsim.ImageType.Scene, False, False
        )]

        # Collect all apples' information
        self.all_apples = []
        self.name_all_apples = []
        for idx in range(101):
            if str(self.drone.simGetObjectPose('tefcefala_LOD1_Blueprint_' + str(idx)).position.x_val) != 'nan':
                self.all_apples.append({'object_name': 'tefcefala_LOD1_Blueprint_' + str(idx),
                                        'pose': self.drone.simGetObjectPose('tefcefala_LOD1_Blueprint_' + str(idx)),
                                        'scale': self.drone.simGetObjectScale('tefcefala_LOD1_Blueprint_' + str(idx))})
                self.name_all_apples.append('tefcefala_LOD1_Blueprint_' + str(idx))

        self.all_apples_obstructed = []
        self.name_all_apples_obstructed = []
        for idx in range(102):
            if str(self.drone.simGetObjectPose('tefcefala_LOD1_obstructed_Blueprint_' + str(idx)).position.x_val) != 'nan':
                self.all_apples_obstructed.append({'object_name': 'tefcefala_LOD1_obstructed_Blueprint_' + str(idx),
                                        'pose': self.drone.simGetObjectPose('tefcefala_LOD1_obstructed_Blueprint_' + str(idx)),
                                        'scale': self.drone.simGetObjectScale('tefcefala_LOD1_obstructed_Blueprint_' + str(idx))})
                self.name_all_apples_obstructed.append('tefcefala_LOD1_obstructed_Blueprint_' + str(idx))

    def __del__(self):
        print('__del__')
        self.drone.reset()

    def _setup_flight(self):
        """Setup drone and take off"""
        print('_setup_flight')
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        time.sleep(0.1)

        self.drone.takeoffAsync().join()

    def _transform_obs(self, responses):
        """
        Process drone's image

        Args:
            responses (responses simGetImages): Response of the simGetImages

        Returns:
            uint8: Drone's image in grayscale or BGR
        """
        #print('transform_obs')

        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)

        # Reshape array (AirSim return flatten image). Default shape=(144, 256, 3)
        img_bgr = img1d.reshape(response.height, response.width, 3)
        try:
            image = Image.fromarray(img_bgr)
        except Exception as e:
            print(e)
            print(f'reponse: {response}')
            print(f'response.height: {response.height}')
            print(f'response.width: {response.width}')
            print(f'Img_bgr: {img_bgr}')
            raise e

        # If grayscale
        if self.image_shape[2] == 1:
            image_gray = image.resize((self.image_shape[0], self.image_shape[1])).convert('L')
            image_gray_final = np.array(image_gray)
            return image_gray_final.reshape([image_gray_final.shape[0], image_gray_final.shape[1], 1])
        else:
            image_color = image.resize((self.image_shape[0], self.image_shape[1]))
            image_color_final = np.array(image_color)
            return image_color_final

    def _get_obs(self):
        """
        Capture drone's camera image and get drone's status

        Returns:
            uint8: Drone's image in grayscale or BGR
        """
        while True:
            responses = self.drone.simGetImages(self.image_request)
            if responses[0].camera_name != '':
                break

        img_transformed = self._transform_obs(responses)
        self.drone_state = self.drone.getMultirotorState()

        self.state['prev_drone_position'] = self.state['drone_position']
        self.state['drone_position'] = self.drone_state.kinematics_estimated.position
        # Move the referential point from drone's middle to cup (hand)
        self.state['drone_position'].x_val += 0.82
        self.state['drone_position'].z_val += 0.16
        self.state['drone_velocity'] = self.drone_state.kinematics_estimated.linear_velocity


        # Object in Unreal that has information about the drone's status (arm collision and grabber)
        status_drone = self.drone.simGetObjectPose('status')
        temp = self.drone.simGetCollisionInfo()
        self.state['drone_collision'] = temp.has_collided
        self.state['code_grab_overlap'] = round(status_drone.position.x_val)
        self.state['code_arm_collision'] = round(status_drone.position.y_val)
        self.state['code_amount_apple'] = self.code_amount_apple

        return img_transformed

    def _do_action(self, action, is_suck_apple, step_length=0.5):
        """
        Move the drone

        Args:
            action (int or tuple[float, float, float]): Movement to the drone do
            step_length (float): Drone's movement velocity, used if the action is an int

        Returns:
            tuple[float, float, float]
                X: Velocity to move the drone coordinate X
                Y: Velocity to move the drone coordinate Y
                Z: Velocity to move the drone coordinate Z
        """
        quad_offset = self._interpret_action(action, step_length)
        self.drone.moveByVelocityAsync(
            quad_offset[0],
            quad_offset[1],
            quad_offset[2],
            0.001,
        ).join()

        if is_suck_apple:
            self.drone.simRunConsoleCommand('ke FlyingPawn GrabApple')

        return quad_offset

    def _compute_reward(self):
        """
        Calculate the reward

        Returns:
            tuple[float, bool]
                reward: Reward of the action done (only for environment 1)
                done: Tag to indicate if the epoch ended
        """
        done = False

        # α = Overlap, v = velocity, β = fail, ρ = next_apple, ∆ = distance
        # reward = (1000 - 1000v)α − 100β + 10ρ + 20∆ − 1
        # ∆ = Dt−1 − Dt
        # Alpha
        self.is_overlapping_normal_apple = 0
        # v
        self.velocity_grab_apple = abs(self.state['drone_velocity'].x_val) + abs(self.state['drone_velocity'].y_val) +\
                                   abs(self.state['drone_velocity'].z_val)
        # Beta
        is_failed = 0

        if self.nstep >= self.nstep_limit or self.state['drone_collision'] or self.state['code_arm_collision'] > 10000:
            is_failed = 1
            done = True
        else:
            if self.state['code_grab_overlap'] >= 11000 and self.state['code_grab_overlap'] <= 11999:
                #print('OVERLAPPING')
                pass
            if self.state['code_grab_overlap'] >= 21000 and self.state['code_grab_overlap'] <= 21999:
                self.is_overlapping_normal_apple = 1
                done = True
            elif self.state['code_grab_overlap'] >= 22000 and self.state['code_grab_overlap'] <= 24999:
                done = True
            # Overlapping green apple
            elif self.state['code_grab_overlap'] >= 12000 and self.state['code_grab_overlap'] <= 12999:
                #done = True
                pass
            # Overlapping obstructed normal apple
            elif self.state['code_grab_overlap'] >= 13000 and self.state['code_grab_overlap'] <= 13999:
                #done = True
                pass
            # Overlapping obstructed green apple
            elif self.state['code_grab_overlap'] >= 14000 and self.state['code_grab_overlap'] <= 14999:
                #done = True
                pass

        reward = None
        if self.environment == 1:
            self.state['apple_position'] = self._get_positions('tefcefala_LOD1_Blueprint_1')
            dist = self._euclidean_distance(self.state['drone_position'], self.state['apple_position'])

            # Delta
            self.diff_distance = self.prev_dist - dist
            self.prev_dist = dist

            reward = self.calculate_reward(is_failed)
        return reward, done

    def calculate_reward(self, is_failed):
        return (1000 - 1000 * self.velocity_grab_apple) * self.is_overlapping_normal_apple - 100 * is_failed + 20 * self.diff_distance #- 1

    def step(self, action, is_suck_apple=False, step_length=0.5, test=0):
        """
        Main method to move drone, collect environment observation and calculate the reward

        Args:
            action (int or tuple[float, float, float]): Movement to the drone to do
            is_suck_apple (bool): Tag to indicate whether to grab the apple
            step_length (float): Drone's movement velocity, used if the action is a list
            test (int): Codes to test functions

        Returns:
            tuple [PIL_Image, float, bool, dict, tuple[float, float, float]]
                obs: Drone's image in grayscale or RGB
                reward: Reward of the action done
                done: Tag to indicate if the epoch ended
                self.state: Information about drone and arm collision and hand overlap
                quad_offset: Velocity to move the drone X, Y and Z
        """
        if test == 888:
            print(self.drone.simRunConsoleCommand('ke FlyingPawn GrabApple'))
        if test == 999:
            print(self.drone.simRunConsoleCommand('ke FlyingPawn ReleaseApple'))

        self.nstep += 1
        quad_offset = self._do_action(action, is_suck_apple, step_length)
        obs = self._get_obs()
        reward, done = self._compute_reward()

        return obs, reward, done, self.state, quad_offset

    def reset(self, is_reset_environment=True, test_tracker=False, dist_y_between_apple=0.075, dist_z_between_apple=0.08):
        """
        Reset environment and drone position

        Returns:
            tuple(PIL_Image, dict):
                PIL_Image: Drone's image in grayscale or RGB
                self.state: Information about drone and arm collision and hand overlap
        """
        print('reset')

        self.release_apple()
        self.drone.reset()
        self.reset_status()

        if is_reset_environment:
            # For environment 1 the "IsAutoGrab" inside "BP_flyingPawn_cup" in Unreal must be False
            if self.environment == 1:
                # Change objective's position
                x = np.random.uniform(1.22, 1.42)
                y = np.random.uniform(-0.64, 0.64)
                # left
                #y = np.random.uniform(-0.64, -0.13)
                # right
                #y = np.random.uniform(0.14, 0.64)

                z = np.random.uniform(-1.70, -2.40)
                # up
                #z = np.random.uniform(-2.20, -2.40)
                # down
                #z = np.random.uniform(-1.70, -1.95)

                # If the apple will be behind drone's arm, change coordinates
                while y > -0.13 and y < 0.13 and z > -2.0:
                    y = np.random.uniform(-0.64, 0.64)
                    z = np.random.uniform(-1.70, -2.40)

                # Destroy apple
                for apple in self.all_apples:
                    if not self.drone.simDestroyObject(apple['object_name']):
                        # Second check
                        time.sleep(0.2)
                        self.drone.simDestroyObject(apple['object_name'])

                # Time to Unreal process destroy
                time.sleep(2)
                # Spawn apple
                for apple in self.all_apples:
                    name = self.drone.simSpawnObject(apple['object_name'], 'tefcefala_LOD1_Blueprint',
                                                     apple['pose'],
                                                     apple['scale'], is_blueprint=True)
                    time.sleep(0.2)
                    # Scale in simSpawnObject does not works!
                    self.drone.simSetObjectScale(name, apple['scale'])
                    if not test_tracker:
                        break

                self.drone.simSetObjectPose('tefcefala_LOD1_Blueprint_1', airsim.Pose(airsim.Vector3r(x, y, z)), True)
                if test_tracker:
                    self.drone.simSetObjectPose('tefcefala_LOD1_Blueprint_2', airsim.Pose(airsim.Vector3r(x, y + dist_y_between_apple, z)), True)
                    self.drone.simSetObjectPose('tefcefala_LOD1_Blueprint_3', airsim.Pose(airsim.Vector3r(x, y - dist_y_between_apple, z)), True)
                    self.drone.simSetObjectPose('tefcefala_LOD1_Blueprint_4', airsim.Pose(airsim.Vector3r(x, y, z + dist_z_between_apple)), True)
                    self.drone.simSetObjectPose('tefcefala_LOD1_Blueprint_5', airsim.Pose(airsim.Vector3r(x, y, z - dist_z_between_apple)), True)
                    self.drone.simSetObjectPose('tefcefala_LOD1_Blueprint_6', airsim.Pose(airsim.Vector3r(x, y + dist_y_between_apple, z + dist_z_between_apple)), True)
                    self.drone.simSetObjectPose('tefcefala_LOD1_Blueprint_7', airsim.Pose(airsim.Vector3r(x, y + dist_y_between_apple, z - dist_z_between_apple)), True)
                    self.drone.simSetObjectPose('tefcefala_LOD1_Blueprint_8', airsim.Pose(airsim.Vector3r(x, y - dist_y_between_apple, z + dist_z_between_apple)), True)
                    self.drone.simSetObjectPose('tefcefala_LOD1_Blueprint_9', airsim.Pose(airsim.Vector3r(x, y - dist_y_between_apple, z - dist_z_between_apple)), True)
            else:
                # Destroy all apples
                for apple in self.all_apples:
                    if not self.drone.simDestroyObject(apple['object_name']):
                        # Second check
                        time.sleep(0.2)
                        self.drone.simDestroyObject(apple['object_name'])
                for apple in self.all_apples_obstructed:
                    if not self.drone.simDestroyObject(apple['object_name']):
                        # Second check
                        time.sleep(0.2)
                        self.drone.simDestroyObject(apple['object_name'])

                # Time to Unreal process destroy
                time.sleep(2)

                # Spawn some apples
                for apple in self.all_apples:
                    if random.randint(0, 100) >= self.threshold_spawn_apple:
                        name = self.drone.simSpawnObject(apple['object_name'], 'tefcefala_LOD1_Blueprint', apple['pose'],
                                                         apple['scale'], is_blueprint=True)
                        
                        # If create with wrong name, delete and create again
                        if name not in self.name_all_apples:
                            time.sleep(0.2)
                            self.drone.simDestroyObject(name)
                            time.sleep(0.2)
                            self.drone.simDestroyObject(apple['object_name'])
                            time.sleep(0.2)
                            name = self.drone.simSpawnObject(apple['object_name'], 'tefcefala_LOD1_Blueprint', apple['pose'],
                                                         apple['scale'], is_blueprint=True)
                        
                        time.sleep(0.2)
                        # Scale in simSpawnObject does not works!
                        self.drone.simSetObjectScale(name, apple['scale'])

                total_apple_obstructed = 0
                for apple in self.all_apples_obstructed:
                    if random.randint(0, 100) >= self.threshold_spawn_apple:
                        total_apple_obstructed += 1
                        name = self.drone.simSpawnObject(apple['object_name'], 'tefcefala_LOD1_obstructed_Blueprint', apple['pose'],
                                                         apple['scale'], is_blueprint=True)
                        
                        # If create with wrong name, delete and create again
                        if name not in self.name_all_apples_obstructed:
                            time.sleep(0.2)
                            self.drone.simDestroyObject(name)
                            time.sleep(0.2)
                            self.drone.simDestroyObject(apple['object_name'])
                            time.sleep(0.2)
                            name = self.drone.simSpawnObject(apple['object_name'], 'tefcefala_LOD1_obstructed_Blueprint', apple['pose'],
                                                         apple['scale'], is_blueprint=True)

                        time.sleep(0.2)
                        # Scale in simSpawnObject does not works!
                        self.drone.simSetObjectScale(name, apple['scale'])

                self.update_status()
                time.sleep(0.3)
                status_drone = self.drone.simGetObjectPose('status')
                # The number is too big for Unreal, so this calculation is necessary
                self.code_amount_apple = round(status_drone.position.z_val)
                self.code_amount_apple = abs(self.code_amount_apple * 100 - (
                    abs(int(str(self.code_amount_apple)[-2:]) - total_apple_obstructed)))

        self._setup_flight()
        self.nstep = 0

        return self._get_obs(), self.state

    def _interpret_action(self, action, step_length=0.5):
        """
        Interpret action to move the drone

        Args:
            action (int or tuple[float, float, float]): Movement to the drone do; list with 3 float elements
            step_length (float): Drone's movement velocity, used if the action is a list

        Returns:
            tuple [float, float, float]
                X: Velocity to move the drone coordinate X
                Y: Velocity to move the drone coordinate Y
                Z: Velocity to move the drone coordinate Z
        """

        # Move drone using predefined speeds
        if isinstance(action, int):
            # Help to normalize movement
            if step_length >= 0.5:
                div = 6
            else:
                div = 1.7

            # Smart exploration
            if action == 800:
                apple_position = self._get_positions('tefcefala_LOD1_Blueprint_1')
                #x = np.random.rand()
                x = 0.1
                if self.state['drone_position'].y_val < apple_position.y_val:
                    y = np.random.rand()
                else:
                    y = -np.random.rand()
                if self.state['drone_position'].z_val < apple_position.z_val:
                    z = np.random.rand()
                else:
                    z = -np.random.rand()
                quad_offset = (x, y, z)

            # Do nothing
            elif action == 0:
                quad_offset = (0., 0., 0.)
            # Grab
            elif action == 1:
                self.drone.simRunConsoleCommand('ke FlyingPawn GrabApple')
                quad_offset = (0., 0., 0.)
            # Forward
            elif action == 2:
                quad_offset = (step_length, 0., -step_length/div)
            # Right
            elif action == 3:
                quad_offset = (0., step_length, -step_length/div)
            # Down
            elif action == 4:
                quad_offset = (0., 0., step_length)
            # Behind
            elif action == 5:
                quad_offset = (-step_length, 0., -step_length/div)
            # Left
            elif action == 6:
                quad_offset = (0., -step_length, -step_length/div)
            # Up
            elif action == 7:
                quad_offset = (0., 0., -step_length)
            else:
                print(f'Action "{action}" invalid!')
                quad_offset = (0., 0., 0.)
        # Move drone using the values in list
        else:
            quad_offset = action

        return quad_offset

    @staticmethod
    def _euclidean_distance(obj1, obj2):
        """
        Euclidean distance between two objects

        Args:
            obj1 (object_position): Object's position in Unreal
            obj2 (object_position): Object's position in Unreal

        Returns:
            float: Euclidean distance between the objects

        """
        p1 = np.array(
            [
                obj1.x_val,
                obj1.y_val,
                obj1.z_val,
            ]
        )
        p2 = np.array(
            [
                obj2.x_val,
                obj2.y_val,
                obj2.z_val,
            ]
        )

        # Euclidean distance
        return np.linalg.norm(p1 - p2)

    def _get_positions(self, name_object):
        """
        Get position of a object in Unreal

        Args:
            name_object: Object's name in Unreal to get position

        Returns:
            object_position: Object's position in Unreal
        """
        # Apple and drone position
        return self.drone.simGetObjectPose(name_object).position

    def move_drone_to_position(self, position, velocity=0.5, type=1):
        if type == 1:
            self.drone.moveToPositionAsync(position[0]/100, position[1]/100, position[2]/100, velocity).join()
        else:
            self.drone.moveToZAsync(position[2]/100, velocity).join()

    def move_drone_by_velocity(self, quad_offset, time=0.001):
        self.drone.moveByVelocityAsync(
            quad_offset[0],
            quad_offset[1],
            quad_offset[2],
            time,
        ).join()

    def reset_status(self):
        """
        Reset environment value status
        """
        self.drone.simRunConsoleCommand('ke FlyingPawn ResetStatus')

    def update_status(self):
        """
        Update environment value status
        """
        self.drone.simRunConsoleCommand('ke FlyingPawn UpdateStatus')

    def release_apple(self):
        """
        Release the apple from the suction cup
        """
        self.drone.simRunConsoleCommand('ke FlyingPawn ReleaseApple')

    def set_auto_grab(self, is_activate):
        """
        Change if the drone will automatically pick the apple when touch in it

        Args:
            is_activate (bool): Whether to pick the apple automatically when touching in it
        """
        if is_activate:
            self.drone.simRunConsoleCommand('ke FlyingPawn ActivateAutoGrab')
        else:
            self.drone.simRunConsoleCommand('ke FlyingPawn DisableAutoGrab')

    def disable_percentage_apple(self):
        """
        All apples not obstructed will always be red (NR)
        """
        self.drone.simRunConsoleCommand('ke FlyingPawn DisablePercentageApple')