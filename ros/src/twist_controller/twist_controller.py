from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter


GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class TwistController(object):
    def __init__(self, wheel_base, steer_ratio, max_lat_accel, max_steer_angle, decel_limit, accel_limit):

        # PID deceleration controller
        self.pid_brake = PID(0.2, 0, 0, mn=0, mx=-decel_limit)
        self.low_pass_brake = LowPassFilter(0.1, 1/50)

        # PID acceleration controller
        self.pid_gas = PID(0.5, 0.5, 0, mn=0, mx=accel_limit)
        self.low_pass_gas = LowPassFilter(0.1, 1 / 50)

        min_speed = 0.5
        self.yaw_controller = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)

    def control(self, current_long_velocity, desired_long_velocity, desired_angular_velocity):

        steer_angle = self.yaw_controller.get_steering(current_long_velocity,
                                                       desired_angular_velocity,
                                                       desired_long_velocity)

        velocity_error = desired_long_velocity - current_long_velocity

        brake_pos = self.pid_brake.step(velocity_error, 1/50)
        brake_pos_filt = self.low_pass_brake.filt(brake_pos)

        gas_pos = self.pid_gas.step(velocity_error, 1/50)
        gas_pos_filt = self.low_pass_gass.filt(gas_pos)

        return gas_pos_filt, brake_pos_filt, steer_angle
