DESC = '''
Helper script to record/examine a rollout's openloop effects (render/ playback/ recover) on an environment\n
  > Examine options:\n
    - Record:   Record an execution. (Useful for kinesthetic demonstrations on hardware)\n
    - Render:   Render back the execution. (sim.forward)\n
    - Playback: Playback the rollout action sequence in openloop (sim.step(a))\n
    - Recover:  Playback actions recovered from the observations \n
  > Render options\n
    - either onscreen, or offscreen, or just rollout without rendering.\n
  > Save options:\n
    - save resulting paths as pickle or as 2D plots\n
USAGE:\n
    $ python examine_rollout.py --env_name door-v0 \n
    $ python examine_rollout.py --env_name door-v0 --rollout_path my_rollouts.pickle --repeat 10 \n
'''

import gym
from myosuite.utils.viz_paths import plot_paths as plotnsave_paths
from myosuite.utils import tensor_utils
import click
import numpy as np
import pickle
import time
import os
import skvideo.io


@click.command(help=DESC)
@click.option('-e', '--env_name', type=str, help='environment to load', required=True)
@click.option('-p', '--rollout_path', type=str, help='absolute path of the rollout', default=None)
@click.option('-m', '--mode', type=click.Choice(['record', 'render', 'playback', 'recover']), help='How to examine rollout', default='playback')
@click.option('-h', '--horizon', type=int, help='Rollout horizon, when mode is record', default=-1)
@click.option('-s', '--seed', type=int, help='seed for generating environment instances', default=123)
@click.option('-n', '--num_repeat', type=int, help='number of repeats for the rollouts', default=1)
@click.option('-r', '--render', type=click.Choice(['onscreen', 'offscreen', 'none']), help='visualize onscreen or offscreen', default='onscreen')
@click.option('-c', '--camera_name', type=str, default=None, help=('Camera name for rendering'))
@click.option('-o', '--output_dir', type=str, default='./', help=('Directory to save the outputs'))
@click.option('-on', '--output_name', type=str, default=None, help=('The name to save the outputs as'))
@click.option('-sp', '--save_paths', type=bool, default=False, help=('Save the rollout paths'))
@click.option('-cp', '--compress_paths', type=bool, default=True, help=('compress paths. Remove obs and env_info/state keys'))
@click.option('-pp', '--plot_paths', type=bool, default=False, help=('2D-plot of individual paths'))
@click.option('-ea', '--env_args', type=str, default=None, help=('env args. E.g. --env_args "{\'is_hardware\':True}"'))
@click.option('-ns', '--noise_scale', type=float, default=0.0, help=('Noise amplitude in randians}"'))

def main(env_name, rollout_path, mode, horizon, seed, num_repeat, render, camera_name, output_dir, output_name, save_paths, compress_paths, plot_paths, env_args, noise_scale):

    # seed and load environments
    np.random.seed(seed)
    env = gym.make(env_name) if env_args==None else gym.make(env_name, **(eval(env_args)))
    env.seed(seed)

    # load paths
    if mode == 'record':
        assert horizon>0, "Rollout horizon must be specified when recording rollout"
        assert output_name is not None, "Specify the name of the recording"
        if save_paths is False:
            print("Warning: Recording is not being saved. Enable save_paths=True to log the recorded path")
        paths = [None,]*num_repeat # empty paths for recordings
    else:
        assert rollout_path is not None, "Rollout path is required for mode:{} ".format(mode)
        paths = pickle.load(open(rollout_path, 'rb'))
        if output_dir == './': # overide the default
            output_dir = os.path.dirname(rollout_path)
        if output_name is None:
            rollout_name = os.path.split(rollout_path)[-1]
            output_name = os.path.splitext(rollout_name)[0]

    # resolve rendering
    if render == 'onscreen':
        env.env.mujoco_render_frames = True
    elif render =='offscreen':
        env.mujoco_render_frames = False
        frame_size=(640,480)
        frames = np.zeros((env.horizon, frame_size[1], frame_size[0], 3), dtype=np.uint8)
    elif render == None:
        env.mujoco_render_frames = False

    # playback paths
    pbk_paths = []
    for i_loop in range(num_repeat):
        print("Starting playback loop:{}".format(i_loop))
        ep_rwd = 0.0
        for i_path, path in enumerate(paths):

            # initialize buffers
            ep_t0 = time.time()
            obs = []
            act = []
            rewards = []
            env_infos = []
            states = []

            # initialize env to the starting position
            if path and "state" in path['env_infos'].keys():
                env.reset(reset_qpos=path['env_infos']['state']['qpos'][0], reset_qvel=path['env_infos']['state']['qvel'][0])
            else:
                env.reset()

            # Rollout
            o = env.get_obs()
            path_horizon = horizon if mode == 'record' else path['actions'].shape[0]
            for i_step in range(path_horizon):

                # Record Execution. Useful for kinesthetic demonstrations on hardware
                if mode=='record':
                    a = env.action_space.sample() # dummy random sample
                    onext, r, d, info = env.step(a) # t ==> t+1

                # Directly create the scene
                elif mode=='render':
                    env.sim.data.qpos[:]= path['env_infos']['state']['qpos'][i_step]
                    env.sim.data.qvel[:]= path['env_infos']['state']['qvel'][i_step]
                    env.sim.forward()
                    env.mj_render()

                    # copy over from exiting path
                    a = path['actions'][i_step]
                    if (i_step+1) < path_horizon:
                        onext = path['observations'][i_step+1]
                        r = path['rewards'][i_step+1]
                        info = {}

                # Apply actions in open loop
                elif mode=='playback':
                    a = path['actions'][i_step]
                    onext, r, d, info = env.step(a) # t ==> t+1

                # Recover actions from states
                elif mode=='recover':
                    # assumes position controls
                    a = path['env_infos']['obs_dict']['qp'][i_step]
                    if noise_scale:
                        a = a +  env.env.np_random.uniform(high=noise_scale, low=-noise_scale, size=len(a)).astype(a.dtype)
                    if env.normalize_act:
                        a = env.robot.normalize_actions(controls=a)
                    onext, r, d, info = env.step(a) # t ==> t+1

                # populate rollout paths
                ep_rwd += r
                act.append(a)
                rewards.append(r)
                if compress_paths:
                    obs.append([]); o = onext # don't save obs
                    del info['state']  # don't save state
                else:
                    obs.append(o); o = onext
                env_infos.append(info)

                # Render offscreen
                if render =='offscreen':
                    curr_frame = env.render_camera_offscreen(
                        sim=env.sim,
                        cameras=[camera_name],
                        width=frame_size[0],
                        height=frame_size[1],
                        device_id=0
                    )
                    frames[i_step,:,:,:] = curr_frame[0]
                    print(i_step, end=', ', flush=True)

            # Create rollout outputs
            pbk_path = dict(observations=np.array(obs),
                actions=np.array(act),
                rewards=np.array(rewards),
                env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
                states=states)
            pbk_paths.append(pbk_path)

            # save offscreen buffers as video
            if render =='offscreen':
                file_name = output_dir + 'rollout' + str(i_path) + ".mp4"
                skvideo.io.vwrite(file_name, np.asarray(frames))
                print("saved", file_name)

            # Finish rollout
            print("-- Finished playback path %d :: Total reward = %3.3f, Total time = %2.3f" % (i_path, ep_rwd, ep_t0-time.time()))

        # Finish loop
        print("Finished playback loop:{}".format(i_loop))

    # Save paths
    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    if save_paths:
        file_name = os.path.join(output_dir, output_name + '{}_paths.pickle'.format(time_stamp))
        pickle.dump(pbk_paths, open(file_name, 'wb'))
        print("Saved: "+file_name)

    # plot paths
    if plot_paths:
        file_name = os.path.join(output_dir, output_name + '{}'.format(time_stamp))
        plotnsave_paths(pbk_paths, env=env, fileName_prefix=file_name)

if __name__ == '__main__':
    main()
