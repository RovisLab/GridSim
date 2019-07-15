from pygame.math import Vector2
from math import tan, radians, degrees, copysign
import random
import numpy as np


class Car:
    def __init__(self, x, y, angle=0.0, length=4, max_steering=30, max_acceleration=30.0, car_tag=None, car_image=None):
        self.position = Vector2(x, y)
        self.velocity = Vector2(0.0, 0.0)
        self.angle = angle
        self.length = length
        self.max_acceleration = max_acceleration
        self.max_steering = max_steering
        self.max_velocity = 15
        self.brake_deceleration = 30
        self.free_deceleration = 5

        self.car_tag = car_tag
        self.car_image = car_image

        self.acceleration = 0.0
        self.steering = 0.0

        self.include_next_lane_mechanic = False
        self.next_lane = None
        self.next_lane_steps = None
        self.next_lane_angles = None

    def reset_car(self, rs_pos_list):
        rand_pos = random.choice(rs_pos_list)
        self.angle = rand_pos[2]
        self.position = (rand_pos[0], rand_pos[1])
        self.velocity.x = 0
        self.velocity.y = 0

    def accelerate(self, dt):
        if self.velocity.x < 0:
            self.acceleration = self.brake_deceleration
        else:
            self.acceleration += 10 * dt

    def brake(self, dt):
        if self.velocity.x > 0:
            self.acceleration = -self.brake_deceleration
        else:
            self.acceleration -= 10 * dt

    def handbrake(self, dt):
        if abs(self.velocity.x) > dt * self.brake_deceleration:
            self.acceleration = -copysign(self.brake_deceleration, self.velocity.x)
        else:
            self.acceleration = -self.velocity.x / dt

    def cruise(self, dt):
        if abs(self.velocity.x) > dt * self.free_deceleration:
            self.acceleration = -copysign(self.free_deceleration, self.velocity.x)
        else:
            if dt != 0:
                self.acceleration = -self.velocity.x / dt

    def steer_right(self, dt):
        self.steering -= 180 * dt

    def steer_left(self, dt):
        self.steering += 180 * dt

    def no_steering(self):
        self.steering = 0

    def update(self, dt):
        self.acceleration = max(-self.max_acceleration, min(self.acceleration, self.max_acceleration))
        self.steering = max(-self.max_steering, min(self.steering, self.max_steering))
        self.velocity += (self.acceleration * dt, 0)
        self.velocity.x = max(-self.max_velocity, min(self.velocity.x, self.max_velocity))

        # this mechanic helps if you have traffic cars that change lane
        if self.include_next_lane_mechanic is True:
            if self.next_lane is not None:
                if self.next_lane - 0.5 <= self.position.x <= self.next_lane + 0.5:
                    self.angle = -90
                    self.position.x = self.next_lane
                    self.next_lane = None
                    self.next_lane_steps = None
                    self.next_lane_angles = None
                else:
                    if self.next_lane_steps is None:
                        if self.next_lane > self.position.x:
                            self.next_lane_steps = np.arange(self.position.x, self.next_lane, 0.1)
                            nr_of_angles = int(self.next_lane_steps.shape[0] / 2)
                            final_angle = -145
                            angle_rate = (final_angle - -90)/nr_of_angles
                            first_angles = np.arange(-90, final_angle, angle_rate)
                            second_angles = np.flip(first_angles)
                            angles = np.concatenate((first_angles, second_angles))
                        else:
                            self.next_lane_steps = np.arange(self.position.x, self.next_lane, -0.1)
                            nr_of_angles = int(self.next_lane_steps.shape[0] / 2)
                            final_angle = -35
                            angle_rate = abs(-90 - final_angle)/nr_of_angles
                            first_angles = np.arange(-90, final_angle, angle_rate)
                            second_angles = np.flip(first_angles)
                            angles = np.concatenate((first_angles, second_angles))

                        self.next_lane_angles = angles

                    if self.next_lane > self.position.x:
                        self.position.x += 0.1
                    elif self.next_lane < self.position.x:
                        self.position.x -= 0.1
                    angle_index = np.where(self.next_lane_steps == self.position.x)
                    if self.next_lane_angles[angle_index].shape[0] != 0:
                        self.angle = self.next_lane_angles[angle_index]
            else:
                self.no_steering()

        if self.steering:
            turning_radius = self.length / tan(radians(self.steering))
            angular_velocity = self.velocity.x / turning_radius
        else:
            angular_velocity = 0

        self.position += self.velocity.rotate(-self.angle) * dt
        self.angle += degrees(angular_velocity) * dt

        if self.angle < 0:
            self.angle = 360 + self.angle
        if self.angle > 360:
            self.angle = self.angle - 360
