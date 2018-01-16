from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter


GAS_DENSITY = 2.858
ONE_MPH = 0.44704
STEP_TIME = 1.0/50


class TwistController(object):
    def __init__(self, wheel_base, steer_ratio, max_lat_accel, max_steer_angle, decel_limit):

        # PID brake controller
        self.pid_brake = PID(1.0, 0, 0, mn=0, mx=-decel_limit)
        self.low_pass_brake = LowPassFilter(0.1, STEP_TIME)

        # PID throttle controller
        self.pid_gas = PID(0.5, 0.0, 0, mn=0, mx=0.3)
        self.low_pass_gas = LowPassFilter(0.1, STEP_TIME)

        min_speed = 0.5
        self.yaw_controller = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)

        # apply brakes if (desired velocity - current velocity) is smaller than this threshold
        self.brake_threshold = -5

    def control(self, current_long_velocity, desired_long_velocity, desired_angular_velocity):

        steer_angle = self.yaw_controller.get_steering(current_long_velocity,
                                                       desired_angular_velocity,
                                                       desired_long_velocity)

        velocity_error = desired_long_velocity - current_long_velocity

        if velocity_error < self.brake_threshold:
            # set gas to 0 and apply brakes
            gas_pos = 0.0
            self.low_pass_gas.reset()

            brake_pos_new = self.pid_brake.step(velocity_error, STEP_TIME)
            brake_pos = self.low_pass_brake.filt(brake_pos_new)
        else:
            # set brakes to 0 and apply gas
            gas_pos_new = self.pid_gas.step(velocity_error, STEP_TIME)
            gas_pos = self.low_pass_gas.filt(gas_pos_new)

            brake_pos = 0.0
            self.low_pass_brake.reset()

        return gas_pos, brake_pos, steer_angle
