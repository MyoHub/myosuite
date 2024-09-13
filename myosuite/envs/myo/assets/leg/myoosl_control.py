""" =================================================
# Copyright (c) MyoSuite Authors
Authors  :: Chun Kwang Tan (cktan.neumove@gmail.com), Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

import numpy as np
import copy

class MyoOSLController:
    def __init__(self, 
                 body_mass,
                 init_state='e_stance',
                 hardware_param=None,
                 n_sets=4,
                 ):
        """
        Initializes the OSL state machine
        Default init state: early stance [e_stance]
        All states: [e_stance, l_stance, e_swing, l_swing]
        - Early stance, Late Stance, Early Swing, Late Swing
        n_sets : Denotes the maximum number of possible sets of state machine variables
        """

        assert init_state in ['e_stance', 'l_stance', 'e_swing', 'l_swing'], "Phase should be : ['e_stance', 'l_stance', 'e_swing', 'l_swing']"
        self.init_state = init_state
        self.n_sets = n_sets

        self.initDefaults(body_mass)

        if hardware_param is not None:
            self.HARDWARE = hardware_param
        
        self.osl_state_list = {}
        for state_name in self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT].keys():
            self.osl_state_list[state_name] = State(state_name, 
                                                    self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT][state_name]['gain'], 
                                                    self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT][state_name]['threshold'])

        # Define state transitions
        self.osl_state_list['e_stance'].next_state = self.osl_state_list['l_stance']
        self.osl_state_list['l_stance'].next_state = self.osl_state_list['e_swing']
        self.osl_state_list['e_swing'].next_state = self.osl_state_list['l_swing']
        self.osl_state_list['l_swing'].next_state = self.osl_state_list['e_stance']

        self.STATE_MACHINE = StateMachine(self.osl_state_list)
        self.STATE_MACHINE.init_machine(init_state)


    def start(self):
        """
        Starts the State Machine
        """
        self.STATE_MACHINE.start()

    def reset(self, init_state=None):
        """
        Resets the State Machine
        Passing in an initial state resets it to a desired state, otherwise it uses the initial state of the OSL controller
        """
        if init_state is not None:
            self.STATE_MACHINE.init_machine(init_state)
            self.init_state = init_state
        else:
            self.STATE_MACHINE.init_machine(self.init_state)

    def update(self, sens_data):
        """
        Updates the controller with new sensor data so state transitions can be checked
        """
        assert all([key in sens_data.keys() for key in ['knee_angle', 'knee_vel', 'load', 'ankle_angle', 'ankle_vel']]), "Missing data, dictionary should contain all of these keys ['knee_angle', 'knee_vel', 'load', 'ankle_angle', 'ankle_vel']"
        self.SENSOR_DATA = sens_data
        self.STATE_MACHINE.update(self.SENSOR_DATA)

    def get_osl_torque(self):
        """
        Internal function to obtain torques from 
        """
        out_joint_list = ['knee', 'ankle']
        return dict(zip( out_joint_list, [self._get_joint_torque(jnt) for jnt in out_joint_list] ))

    def change_osl_mode(self, mode=0):
        """
        In the case of multiple control gains for the OSL, this function can be used to switch between different sets of control gains
        """
        assert mode < self.n_sets # Ensure that no. of parameter sets do not exceed fixed value
        self.OSL_PARAM_SELECT = mode
        self._update_param_to_state_machine()

    def set_osl_param_batch(self, params, mode=0):
        """
        A batch method to upload parameters for the State Machine
        States: ['e_stance', 'l_stance', 'e_swing', 'l_swing']
        Parameter type: ['knee', 'ankle', 'threshold']
        Parameters: ['knee_stiffness', 'knee_damping', 'ankle_stiffness', 'ankle_damping', 'load', 'knee_angle', 'knee_vel', 'ankle_angle']
        """
        assert mode < self.n_sets # Ensure that no. of parameter sets do not exceed fixed value

        phase_list = ['e_stance', 'l_stance', 'e_swing', 'l_swing']
        joint_list = ['knee', 'ankle', 'threshold']
        idx = 0

        if isinstance(params, np.ndarray):
            for phase in phase_list:
                for jnt_arg in joint_list:
                    for key in self.OSL_PARAM_LIST[mode][phase][jnt_arg].keys():
                        self.OSL_PARAM_LIST[mode][phase][jnt_arg][key] = params[idx]
                        idx += 1

        elif isinstance(params, dict):
            self.OSL_PARAM_LIST[mode] = copy.deepcopy(params)

    def set_osl_param(self, phase_name, param_type, gain, value, mode=0):
        """
        Function to set individual parameters of the OSL leg
        """
        assert phase_name in ['e_stance', 'l_stance', 'e_swing', 'l_swing'], f"Phase should be : {['e_stance', 'l_stance', 'e_swing', 'l_swing']}"
        assert param_type in ['gain', 'threshold'], f"Type should be : {['gain', 'threshold']}"
        assert gain in ['knee_stiffness', 'knee_damping', 'ankle_stiffness', 'ankle_damping', 'load', 'knee_angle', 'knee_vel', 'ankle_angle'], f"Gains should be : {['knee_stiffness', 'knee_damping', 'ankle_stiffness', 'ankle_damping', 'load', 'knee_angle', 'knee_vel', 'ankle_angle']}"

        self.OSL_PARAM_LIST[mode][phase_name][param_type][gain] = value

    def set_motor_param(self, joint, act_param):
        """
        Function to set hardware parameters of the actuators
        """
        assert joint in ['knee', 'ankle'], f"Joint should be : {['knee', 'ankle']}"
        assert act_param in ['gear_ratio', 'peak_torque', 'control_range'], f"Actuator parameter should be : {['gear_ratio', 'peak_torque', 'control_range']}"

        self.HARDWARE[joint][act_param] = act_param

    def _update_param_to_state_machine(self):
        "Internal function to update gain paramters into the State Machine"
        # Hidden function, not to be used directly
        self.STATE_MACHINE.update_state_variables(self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT])
        
    def _get_joint_torque(self, joint):
        # Hidden function, not to be used directly
        "Internal function to calculate commanded torques for each joint"
        if joint not in ['knee', 'ankle']:
            print(f"Non-existant joint. Can only be either 'knee' or 'ankle'")
            raise Exception

        state_params = self.STATE_MACHINE.get_current_state.get_variables()

        K = state_params[f"{joint}_stiffness"]
        B = state_params[f"{joint}_damping"]
        theta = state_params[f"{joint}_target_angle"]

        peak_torque = self.HARDWARE[joint]['peak_torque']

        T = np.clip( K*(theta - self.SENSOR_DATA[f"{joint}_angle"]) - B*(self.SENSOR_DATA[f"{joint}_vel"]),
                     -1*peak_torque, peak_torque)

        return T

    def initDefaults(self, body_mass):
        """
        Initialization functions for body weight and default State Machine variables
        """
        self.GRAVITY = 9.81
        self.BODY_MASS = body_mass
        self.BODY_WEIGHT = self.BODY_MASS * self.GRAVITY

        self.SENSOR_DATA = {}
        self.SENSOR_DATA['knee_angle'] = 0
        self.SENSOR_DATA['knee_vel'] = 0
        self.SENSOR_DATA['ankle_angle'] = 0
        self.SENSOR_DATA['ankle_vel'] = 0
        self.SENSOR_DATA['load'] = 0
        
        self.HARDWARE = {}
        self.HARDWARE['knee'] = {}
        self.HARDWARE['knee']['gear_ratio'] = 49.4
        self.HARDWARE['knee']['peak_torque'] = 142.272
        self.HARDWARE['knee']['control_range'] = 2.88
        self.HARDWARE['ankle'] = {}
        self.HARDWARE['ankle']['gear_ratio'] = 58.4
        self.HARDWARE['ankle']['peak_torque'] = 168.192
        self.HARDWARE['ankle']['control_range'] = 2.88
        
        temp_dict = {}
        temp_dict['e_stance'] = {}
        temp_dict['e_stance']['gain'] = {}
        temp_dict['e_stance']['threshold'] = {}
        temp_dict['e_stance']['gain']['knee_stiffness'] = 99.372
        temp_dict['e_stance']['gain']['knee_damping'] = 3.180
        temp_dict['e_stance']['gain']['knee_target_angle'] = np.deg2rad(5)
        temp_dict['e_stance']['gain']['ankle_stiffness'] = 19.874
        temp_dict['e_stance']['gain']['ankle_damping'] = 0
        temp_dict['e_stance']['gain']['ankle_target_angle'] = np.deg2rad(-2)
        temp_dict['e_stance']['threshold']['load'] = (0.25 * self.BODY_WEIGHT, 'above')
        temp_dict['e_stance']['threshold']['ankle_angle'] = (np.deg2rad(6), 'above')

        temp_dict['l_stance'] = {}
        temp_dict['l_stance']['gain'] = {}
        temp_dict['l_stance']['threshold'] = {}
        temp_dict['l_stance']['gain']['knee_stiffness'] = 99.372
        temp_dict['l_stance']['gain']['knee_damping'] = 1.272
        temp_dict['l_stance']['gain']['knee_target_angle'] = np.deg2rad(8)
        temp_dict['l_stance']['gain']['ankle_stiffness'] = 79.498
        temp_dict['l_stance']['gain']['ankle_damping'] = 0.063
        temp_dict['l_stance']['gain']['ankle_target_angle'] = np.deg2rad(-20)
        temp_dict['l_stance']['threshold']['load'] = (0.15 * self.BODY_WEIGHT, 'below')

        temp_dict['e_swing'] = {}
        temp_dict['e_swing']['gain'] = {}
        temp_dict['e_swing']['threshold'] = {}
        temp_dict['e_swing']['gain']['knee_stiffness'] = 39.749
        temp_dict['e_swing']['gain']['knee_damping'] = 0.063
        temp_dict['e_swing']['gain']['knee_target_angle'] = np.deg2rad(60)
        temp_dict['e_swing']['gain']['ankle_stiffness'] = 7.949
        temp_dict['e_swing']['gain']['ankle_damping'] = 0
        temp_dict['e_swing']['gain']['ankle_target_angle'] = np.deg2rad(25)
        temp_dict['e_swing']['threshold']['knee_angle'] = (np.deg2rad(50), 'above')
        temp_dict['e_swing']['threshold']['knee_vel'] = (np.deg2rad(3), 'below')

        temp_dict['l_swing'] = {}
        temp_dict['l_swing']['gain'] = {}
        temp_dict['l_swing']['threshold'] = {}
        temp_dict['l_swing']['gain']['knee_stiffness'] = 15.899
        temp_dict['l_swing']['gain']['knee_damping'] = 3.816
        temp_dict['l_swing']['gain']['knee_target_angle'] = np.deg2rad(5)
        temp_dict['l_swing']['gain']['ankle_stiffness'] = 7.949
        temp_dict['l_swing']['gain']['ankle_damping'] = 0
        temp_dict['l_swing']['gain']['ankle_target_angle'] = np.deg2rad(15)
        temp_dict['l_swing']['threshold']['load'] = (0.4 * self.BODY_WEIGHT, 'above')
        temp_dict['l_swing']['threshold']['knee_angle'] = (np.deg2rad(30), 'below')

        self.OSL_PARAM_SELECT = 0
        self.OSL_PARAM_LIST = {}
        for idx in np.arange(self.n_sets):
            self.OSL_PARAM_LIST[idx] = {}
            self.OSL_PARAM_LIST[idx] = copy.deepcopy(temp_dict)

    @property
    def getOSLparam(self):
        return copy.deepcopy(self.OSL_PARAM_LIST)

    
class State:
    def __init__(self, name, state_variables: dict, thresholds: dict, next_state=None):
        """
        States expect
        - State variables in dictionary form
        - Threshold for state transitions in dictionary form, with tuples as (threshold value, compare), where compare in ['above', 'below'].
            - 'above': Checks if a given sensor value is ABOVE the threshold
            - 'below': Checks if a given sensor value is BELOW the threshold
        - next_state: Object reference to the next state
        """
        self.name = name
        self.thresholds = thresholds  # Dictionary of variable thresholds and comparison types
        self.state_variables = state_variables
        self.next_state = next_state

    def check_transition(self, sens_data):
        """
        Checks if a threshold has been reach for state transitions
        """
        for key, (threshold, condition) in self.thresholds.items():
            if condition == "above" and sens_data[key] > threshold:
                return self.next_state
            elif condition == "below" and sens_data[key] < threshold:
                return self.next_state
        return self

    def get_name(self):
        """
        Getter: Name
        """
        return self.name

    def get_variables(self):
        """
        Getter: State variables
        """
        return copy.deepcopy(self.state_variables)
    
    def get_thresholds(self):
        """
        Getter: Thresholds for state transitions
        """
        return copy.deepcopy(self.thresholds)

    def set_variables(self, new_variables: dict):
        """
        Setter: Sets state variables via a dictionary
        """
        self.state_variables = copy.deepcopy(new_variables)

    def set_thresholds(self, new_thresholds: dict):
        """
        Setter: Sets state transition thresholds via a dictionary
        """
        self.thresholds = copy.deepcopy(new_thresholds)
    
class StateMachine:
    def __init__(self, states):
        """
        State Machine for OSL impedance controller
        """
        self.states = states
        self.current_state = None
        self.running = False

    def init_machine(self, initial_state):
        """
        Initializes the state machine with an initial state
        """
        assert initial_state in self.states.keys(), f"No state named {initial_state}. Should be {self.states.keys()}"
        self.current_state = self.states[initial_state]

    def start(self):
        """
        Starts the state machine
        """
        self.running = True

    def stop(self):
        """
        Stops the state machine
        """
        self.running = False

    def update(self, sens_data):
        """
        Updates the State Machine with environment sensor data and checks for state transitions
        """
        if self.running and self.current_state is not None:
            self.current_state = self.current_state.check_transition(sens_data)

    def update_state_variables(self, variable_dict: dict):
        """
        Update all the state variables in the state list
        States makes a copy of the dictionary to remove references
        """
        for state_name in self.states.keys():
            self.states[state_name].set_variables(variable_dict[state_name]['gain'])
            self.states[state_name].set_thresholds(variable_dict[state_name]['threshold'])
    
    @property
    def is_running(self):
        """
        Boolean to check if State Machine is running
        """
        return self.is_running

    @property
    def get_current_state(self):
        """
        Get current state from the State Machine
        """
        if self.running:
            return copy.deepcopy(self.current_state)
        else:
            return "Not running"

