
import numpy as np
import copy

from opensourceleg.control.state_machine import Event, State, StateMachine
from opensourceleg.osl import OpenSourceLeg


class MyoOSLStateMachine():

    """
    OSL Related
    """
    def __init__(self, 
                 body_mass,
                 init_state='e_stance',
                 hardware_param=None,
                 debug=False,
                 ):
        """
        Initializes the OSL state machine
        Default init state: early stance [e_stance]
        All states: [e_stance, l_stance, e_swing, l_swing]
        - Early stance, Late Stance, Early Swing, Late Swing
        """
        assert init_state in ['e_stance', 'l_stance', 'e_swing', 'l_swing'], "Phase should be : ['e_stance', 'l_stance', 'e_swing', 'l_swing']"

        self.initDefaults()
        self.BODY_MASS = body_mass
        self.BODY_WEIGHT = self.BODY_MASS * self.GRAVITY
        self.DEBUG_MODE = debug

        if hardware_param is not None:
            self.HARDWARE = hardware_param

        self.OSL = OpenSourceLeg(frequency=200)
        self.OSL.add_joint(name="knee", gear_ratio=self.HARDWARE['knee']['gear_ratio'], offline_mode=True)
        self.OSL.add_joint(name="ankle", gear_ratio=self.HARDWARE['ankle']['gear_ratio'], offline_mode=True)

        self.FSM = self.build_state_machine(init_state)

    def start(self):
        """
        Starts the state machine
        """
        self.FSM.start()

    def reset(self, init_state, body_mass=80):
        """
        Rebuilds the state machine with the specified initial state
        """
        if body_mass is not None:
            self.BODY_MASS = body_mass
            self.BODY_WEIGHT = self.BODY_MASS * self.GRAVITY
        
        self.FSM = self.build_state_machine(init_state)

    def update(self, sens_data):
        """
        Updates sensory data for OSL leg
        Expects a dictionary of sensor values
        """
        assert all([key in sens_data.keys() for key in ['knee_angle', 'knee_vel', 'load_cell', 'ankle_angle', 'ankle_vel']]), "Missing data, dictionary should contain all of these keys ['knee_angle', 'knee_vel', 'load_cell', 'ankle_angle', 'ankle_vel']"
        self.SENSOR_DATA = sens_data

        print()

        self.FSM.update()

    def get_torques(self):
        out_joint_list = ['knee', 'ankle']
        return dict(zip( out_joint_list, [self._get_osl_torque(jnt) for jnt in out_joint_list] ))

    def build_state_machine(self, init_state: str):

        early_stance = State(name="e_stance")
        late_stance = State(name="l_stance")
        early_swing = State(name="e_swing")
        late_swing = State(name="l_swing")
        self.OSL_STATE_LIST = [early_stance, late_stance, early_swing, late_swing]

        early_stance.set_knee_impedance_paramters(
            theta=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['e_stance']['knee']['target_angle'], 
            k=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['e_stance']['knee']['stiffness'], 
            b=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['e_stance']['knee']['damping']
        )
        early_stance.make_knee_active()
        early_stance.set_ankle_impedance_paramters(
            theta=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['e_stance']['ankle']['target_angle'], 
            k=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['e_stance']['ankle']['stiffness'], 
            b=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['e_stance']['ankle']['damping']
        )
        early_stance.make_ankle_active()

        late_stance.set_knee_impedance_paramters(
            theta=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['l_stance']['knee']['target_angle'], 
            k=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['l_stance']['knee']['stiffness'], 
            b=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['l_stance']['knee']['damping']
        )
        late_stance.make_knee_active()
        late_stance.set_ankle_impedance_paramters(
            theta=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['l_stance']['ankle']['target_angle'], 
            k=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['l_stance']['ankle']['stiffness'], 
            b=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['l_stance']['ankle']['damping']
        )
        late_stance.make_ankle_active()

        early_swing.set_knee_impedance_paramters(
            theta=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['e_swing']['knee']['target_angle'], 
            k=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['e_swing']['knee']['stiffness'], 
            b=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['e_swing']['knee']['damping']
        )
        early_swing.make_knee_active()
        early_swing.set_ankle_impedance_paramters(
            theta=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['e_swing']['ankle']['target_angle'], 
            k=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['e_swing']['ankle']['stiffness'], 
            b=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['e_swing']['ankle']['damping']
        )
        early_swing.make_ankle_active()

        late_swing.set_knee_impedance_paramters(
            theta=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['l_swing']['knee']['target_angle'], 
            k=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['l_swing']['knee']['stiffness'], 
            b=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['l_swing']['knee']['damping']
        )
        late_swing.make_knee_active()
        late_swing.set_ankle_impedance_paramters(
            theta=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['l_swing']['ankle']['target_angle'], 
            k=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['l_swing']['ankle']['stiffness'], 
            b=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['l_swing']['ankle']['damping']
        )
        late_swing.make_ankle_active()

        foot_flat = Event(name="foot_flat")
        heel_off = Event(name="heel_off")
        toe_off = Event(name="toe_off")
        pre_heel_strike = Event(name="pre_heel_strike")
        heel_strike = Event(name="heel_strike")

        fsm = StateMachine(osl=self.OSL, spoof=self.DEBUG_MODE)

        for item in self.OSL_STATE_LIST:

            if self.DEBUG_MODE:
                item.set_minimum_time_spent_in_state(5)

            if item.name == init_state:
                fsm.add_state(state=item, initial_state=True)
            else:
                fsm.add_state(state=item)

        fsm.add_event(event=foot_flat)
        fsm.add_event(event=heel_off)
        fsm.add_event(event=toe_off)
        fsm.add_event(event=pre_heel_strike)
        fsm.add_event(event=heel_strike)

        # Callback functions need to be implemented as nested functions
        def estance_to_lstance(osl: OpenSourceLeg) -> bool:
            """
            Transition from early stance to late stance when the loadcell
            reads a force greater than a threshold.
            """
            load_cell = self.SENSOR_DATA['load_cell']
            ankle_pos = self.SENSOR_DATA['ankle_angle']

            print(f"Loadcell: {load_cell} vs Weight threshold: {self.BODY_WEIGHT * self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['e_stance']['threshold']['load']}")
            print(f"Ankle: {ankle_pos} > {self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['e_stance']['threshold']['ankle_angle']}")


            return bool(
                load_cell > self.BODY_WEIGHT * self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['e_stance']['threshold']['load']
                and ankle_pos > self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['e_stance']['threshold']['ankle_angle']
            )

        def lstance_to_eswing(osl: OpenSourceLeg) -> bool:
            """
            Transition from late stance to early swing when the loadcell
            reads a force less than a threshold.
            """
            load_cell = self.SENSOR_DATA['load_cell']

            return bool(load_cell < self.BODY_WEIGHT * self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['l_stance']['threshold']['load'])

        def eswing_to_lswing(osl: OpenSourceLeg) -> bool:
            """
            Transition from early swing to late swing when the knee angle
            is greater than a threshold and the knee velocity is less than
            a threshold.
            """
            
            knee_pos = self.SENSOR_DATA['knee_angle']
            knee_vel = self.SENSOR_DATA['knee_vel']

            return bool(
                knee_pos > self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['e_swing']['threshold']['knee_angle']
                and knee_vel < self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['e_swing']['threshold']['knee_vel']
            )

        def lswing_to_estance(osl: OpenSourceLeg) -> bool:
            """
            Transition from late swing to early stance when the loadcell
            reads a force greater than a threshold or the knee angle is
            less than a threshold.
            """

            knee_pos = self.SENSOR_DATA['knee_angle']
            load_cell = self.SENSOR_DATA['load_cell']

            return bool(
                load_cell > self.BODY_WEIGHT * self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['l_swing']['threshold']['load']
                or knee_pos < self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['l_swing']['threshold']['knee_angle']
            )

        fsm.add_transition(
            source=early_stance,
            destination=late_stance,
            event=foot_flat,
            callback=estance_to_lstance,
        )
        fsm.add_transition(
            source=late_stance,
            destination=early_swing,
            event=heel_off,
            callback=lstance_to_eswing,
        )
        fsm.add_transition(
            source=early_swing,
            destination=late_swing,
            event=toe_off,
            callback=eswing_to_lswing,
        )
        fsm.add_transition(
            source=late_swing,
            destination=early_stance,
            event=heel_strike,
            callback=lswing_to_estance,
        )
        
        return fsm

    def _get_osl_torque(self, joint):
        if joint not in ['knee', 'ankle']:
            print(f"Non-existant joint. Can only be either 'knee' or 'ankle'")
            raise Exception

        K = eval(f"self.FSM.current_state.{joint}_stiffness")
        B = eval(f"self.FSM.current_state.{joint}_damping")
        theta = (eval(f"self.FSM.current_state.{joint}_theta"))
        peak_torque = self.HARDWARE[joint]['peak_torque']

        T = np.clip( K*(theta - self.SENSOR_DATA[f"{joint}_angle"]) - B*(self.SENSOR_DATA[f"{joint}_vel"]),
                     -1*peak_torque, peak_torque)

        return T

    def set_osl_params_batch(self, params, mode=0):

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

    def set_state_machine_param(self, phase_name, param_type, gain, value, mode=0):

        assert phase_name in ['e_stance', 'l_stance', 'e_swing', 'l_swing'], f"Phase should be : {['e_stance', 'l_stance', 'e_swing', 'l_swing']}"
        assert param_type in ['knee', 'ankle', 'threshold'], f"Type should be : {['knee', 'ankle', 'threshold']}"
        assert gain in ['stiffness', 'damping', 'load', 'knee_angle', 'knee_vel', 'ankle_angle'], f"Gains should be : {['stiffness', 'damping', 'load', 'knee_angle', 'knee_vel', 'ankle_angle']}"

        self.OSL_PARAM_LIST[mode][phase_name][param_type][gain] = value

    def set_motor_param(self, joint, act_param):

        assert joint in ['knee', 'ankle'], f"Joint should be : {['knee', 'ankle']}"
        assert act_param in ['gear_ratio', 'peak_torque', 'control_range'], f"Actuator parameter should be : {['gear_ratio', 'peak_torque', 'control_range']}"

        self.HARDWARE[joint][act_param] = act_param

    def set_osl_mode(self, mode=0):
        self.OSL_PARAM_SELECT = np.clip(mode, 0, 2)
        self.update_param_to_osl()

    def update_param_to_osl(self):
        """
        Updates the currently selected OSL parameter to the OSL leg state machine
        """
        for item in self.OSL_STATE_LIST:
            item.set_knee_impedance_paramters(
                theta = self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT][item.name]['knee']['target_angle'],
                k = self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT][item.name]['knee']['stiffness'],
                b = self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT][item.name]['knee']['damping'],
            )
            item.set_ankle_impedance_paramters(
                theta=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT][item.name]['ankle']['target_angle'],
                k = self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT][item.name]['ankle']['stiffness'],
                b = self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT][item.name]['ankle']['damping'],
            )

    def initDefaults(self):
        self.GRAVITY = 9.81

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
        temp_dict['e_stance']['knee'] = {}
        temp_dict['e_stance']['ankle'] = {}
        temp_dict['e_stance']['threshold'] = {}
        temp_dict['e_stance']['knee']['stiffness'] = 99.372
        temp_dict['e_stance']['knee']['damping'] = 3.180
        temp_dict['e_stance']['knee']['target_angle'] = np.deg2rad(5)
        temp_dict['e_stance']['ankle']['stiffness'] = 19.874
        temp_dict['e_stance']['ankle']['damping'] = 0
        temp_dict['e_stance']['ankle']['target_angle'] = np.deg2rad(-2)
        temp_dict['e_stance']['threshold']['load'] = 0.25
        temp_dict['e_stance']['threshold']['ankle_angle'] = np.deg2rad(6)

        temp_dict['l_stance'] = {}
        temp_dict['l_stance']['knee'] = {}
        temp_dict['l_stance']['ankle'] = {}
        temp_dict['l_stance']['threshold'] = {}
        temp_dict['l_stance']['knee']['stiffness'] = 99.372
        temp_dict['l_stance']['knee']['damping'] = 1.272
        temp_dict['l_stance']['knee']['target_angle'] = np.deg2rad(8)
        temp_dict['l_stance']['ankle']['stiffness'] = 79.498
        temp_dict['l_stance']['ankle']['damping'] = 0.063
        temp_dict['l_stance']['ankle']['target_angle'] = np.deg2rad(-20)
        temp_dict['l_stance']['threshold']['load'] = 0.15

        temp_dict['e_swing'] = {}
        temp_dict['e_swing']['knee'] = {}
        temp_dict['e_swing']['ankle'] = {}
        temp_dict['e_swing']['threshold'] = {}
        temp_dict['e_swing']['knee']['stiffness'] = 39.749
        temp_dict['e_swing']['knee']['damping'] = 0.063
        temp_dict['e_swing']['knee']['target_angle'] = np.deg2rad(60)
        temp_dict['e_swing']['ankle']['stiffness'] = 7.949
        temp_dict['e_swing']['ankle']['damping'] = 0
        temp_dict['e_swing']['ankle']['target_angle'] = np.deg2rad(25)
        temp_dict['e_swing']['threshold']['knee_angle'] = np.deg2rad(50)
        temp_dict['e_swing']['threshold']['knee_vel'] = np.deg2rad(3)

        temp_dict['l_swing'] = {}
        temp_dict['l_swing']['knee'] = {}
        temp_dict['l_swing']['ankle'] = {}
        temp_dict['l_swing']['threshold'] = {}
        temp_dict['l_swing']['knee']['stiffness'] = 15.899
        temp_dict['l_swing']['knee']['damping'] = 3.816
        temp_dict['l_swing']['knee']['target_angle'] = np.deg2rad(5)
        temp_dict['l_swing']['ankle']['stiffness'] = 7.949
        temp_dict['l_swing']['ankle']['damping'] = 0
        temp_dict['l_swing']['ankle']['target_angle'] = np.deg2rad(15)
        temp_dict['l_swing']['threshold']['load'] = 0.4
        temp_dict['l_swing']['threshold']['knee_angle'] = np.deg2rad(30)

        self.OSL_PARAM_SELECT = 0
        self.OSL_PARAM_LIST = []
        for idx in np.arange(3):
            self.OSL_PARAM_LIST.append(copy.deepcopy(temp_dict))

    @property
    def getOSLparam(self):
        return copy.deepcopy(self.OSL_PARAM_LIST)