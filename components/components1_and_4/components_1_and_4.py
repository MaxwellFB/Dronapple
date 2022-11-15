"""
Component 1 - Approach the tree
Component 4 - Move to pantry
"""
import time


class Component1_4():
    def __init__(self, environment):
        self.environment = environment
        # ## Information about new position drone when do not detect apple to be grabbed in current position ## #
        # Next movement is to up or down
        self.is_moving_up = True
        # Drone position limit
        # self.y_left_limit_start_position_drone = -85
        self.z_down_limit_start_position_drone = -35
        self.limit_time_up_down = 7
        self.count_time_up_down = 0
        self.limit_time_left = 1
        self.count_time_left = 0
        self.change_position_velocity = 0.5
        # How many to move the drone to the new position
        # Left
        self.change_position_y = 85
        # Up and down
        self.change_position_z = 35
        self.current_start_position_drone = [0, 0, 0]
        self.z_safe_position = -80
        self.apple_deposit_position = (-170, -61, -70)

    def start_epoch(self, start_drone_position_x):
        self.is_moving_up = True
        self.count_time_up_down = 0
        self.count_time_left = 0

        self.current_start_position_drone = [start_drone_position_x, 0,
                                             self.z_down_limit_start_position_drone]
        self.environment.move_drone_to_position(self.current_start_position_drone,
                                                velocity=self.change_position_velocity)

    def apple_collected(self, drone_position_z):
        # Retreat
        self.environment.move_drone_by_velocity((-0.5, 0, 0), 1)
        # If drone be down safe position, move to safe position
        if drone_position_z * 100 > self.z_safe_position:
            self.environment.move_drone_to_position(
                [self.current_start_position_drone[0], self.current_start_position_drone[1], self.z_safe_position],
                velocity=self.change_position_velocity)
        else:
            self.environment.move_drone_to_position(
                [self.current_start_position_drone[0], self.current_start_position_drone[1], self.z_safe_position],
                velocity=1)

        # #### Component 4
        # Move drone to deposit position and release apple
        self.environment.move_drone_to_position(self.apple_deposit_position, velocity=1)
        self.environment.release_apple()

        # #### Component 1
        # If last position was below safe position, move to the safe position, otherwise move
        # direct to the last position
        if drone_position_z * 100 > self.z_safe_position + 10:
            self.environment.move_drone_to_position([self.current_start_position_drone[0],
                                                     self.current_start_position_drone[1],
                                                     self.z_safe_position], velocity=0.5)
        else:
            self.environment.move_drone_to_position(self.current_start_position_drone,
                                                    velocity=self.change_position_velocity)
        time.sleep(5)
        self.environment.move_drone_to_position(self.current_start_position_drone,
                                                velocity=self.change_position_velocity, type=2)

    def collision(self, drone_position_x=None):
        # Retreat according to the drone_position_x
        if drone_position_x is not None:
            # Retreat
            self.environment.move_drone_by_velocity((-(drone_position_x - 0.82), 0, 0), 1)
        # Retreat a little
        else:
            self.environment.move_drone_by_velocity((-0.5, 0, 0), 1)
            time.sleep(0.5)

        if self.change_position_velocity == 0.5:
            # Up or down
            self.environment.move_drone_to_position(self.current_start_position_drone,
                                                    velocity=self.change_position_velocity, type=2)
        else:
            # Left
            self.environment.move_drone_to_position(self.current_start_position_drone,
                                                    velocity=self.change_position_velocity)

    def change_position(self):
        if self.change_position_velocity == 0.5:
            # Up or down
            self.environment.move_drone_to_position(self.current_start_position_drone,
                                                    velocity=self.change_position_velocity, type=2)
        else:
            # Left
            self.environment.move_drone_to_position(self.current_start_position_drone,
                                                    velocity=self.change_position_velocity)

        self.change_position_velocity = 0.5

    def update_drone_next_position(self):
        """
        Update to the next drone position

        Returns:
            Bool: Whether is possible to move the drone to the next position
        """
        if self.is_moving_up:
            # Up
            if self.count_time_up_down < self.limit_time_up_down:
                self.current_start_position_drone[2] -= self.change_position_z
                self.count_time_up_down += 1
                self.change_position_velocity = 0.5
            else:
                # Left
                self.is_moving_up = False
                if self.count_time_left < self.limit_time_left:
                    self.current_start_position_drone[1] -= self.change_position_y
                    self.count_time_up_down = 0
                    self.count_time_left += 1
                    self.change_position_velocity = 0.2
                # End position
                else:
                    return False
        else:
            # Down
            if self.count_time_up_down < self.limit_time_up_down:
                self.current_start_position_drone[2] += self.change_position_z
                self.count_time_up_down += 1
                self.change_position_velocity = 0.5
            else:
                # Left
                self.is_moving_up = True
                if self.count_time_left < self.limit_time_left:
                    self.current_start_position_drone[1] -= self.change_position_y
                    self.count_time_up_down = 0
                    self.count_time_left += 1
                    self.change_position_velocity = 0.2
                # End position
                else:
                    return False
        return True
