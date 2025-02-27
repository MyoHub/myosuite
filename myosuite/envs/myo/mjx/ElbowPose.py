import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=100)
from datetime import datetime
import functools
import jax
from jax import numpy as jp
from matplotlib import pyplot as plt
import mujoco
from mujoco import mjx
from brax import envs
from brax.envs.base import Env, PipelineEnv, State
from brax.training.agents.ppo import train as ppo
from brax.io import mjcf, model
import mujoco.viewer

print(f"Current backend: {jax.default_backend()}")

class Elbow(PipelineEnv):

  def __init__(
      self,
      angle_reward_weight=2.5,
      ctrl_cost_weight=0.1,
      healthy_angle_range=(0, 2.1),
      reset_noise_scale=0, #1e-1,
      is_msk=True,
      qpos0=0.,
      **kwargs,
  ):
    # path = rf"../assets/elbow/myoelbow_1dof{6 if is_msk else 0}muscles_mjx.xml"
    path = rf"../assets/elbow/myoelbow_1dof{2 if is_msk else 0}muscles_mjx.xml"
    mj_model = mujoco.MjModel.from_xml_path(path)
    
    # Solver params: These are seemingly still stable on CPU mujoco,
    # but could be unstable in MJX, need to verify.
    mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
    mj_model.opt.iterations = 6
    mj_model.opt.ls_iterations = 6
    mj_model.opt.disableflags = mj_model.opt.disableflags | mjx.DisableBit.EULERDAMP

    sys = mjcf.load_model(mj_model)

    physics_steps_per_control_step = 5
    kwargs['n_frames'] = kwargs.get(
        'n_frames', physics_steps_per_control_step)
    kwargs['backend'] = 'mjx'

    super().__init__(sys, **kwargs)

    self._angle_reward_weight = angle_reward_weight
    self._ctrl_cost_weight = ctrl_cost_weight
    self._healthy_angle_range = healthy_angle_range
    self._reset_noise_scale = reset_noise_scale
    self._qpos0 =qpos0

  def reset(self, rng: jp.ndarray) -> State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2, rng3 = jax.random.split(rng, 4)
    
    low, hi = -self._reset_noise_scale, self._reset_noise_scale
    qpos =jax.random.uniform(
        rng1, (self.sys.nq,), minval=low, maxval=hi
    ) + self._qpos0  # + self.sys.qpos0
    qvel = jax.random.uniform(
        rng2, (self.sys.nv,), minval=low, maxval=hi
    )

    target_angle = jax.random.uniform(
        rng3, (1,), minval=self._healthy_angle_range[0], maxval=self._healthy_angle_range[1]
        )

    # We store the target angle in the info, can't store it as an instance variable,
    # as it has to be determined in a parallelized manner 
    info = {'rng': rng, 'target_angle': target_angle}

    data = self.pipeline_init(qpos, qvel)

    obs = self._get_obs(data, jp.zeros(self.sys.nu), info)
    reward, done, zero = jp.zeros(3)
    metrics = {
        'angle_reward': zero,
        'reward_quadctrl': zero,
    }
    return State(data, obs, reward, done, metrics, info)

  def step(self, state: State, action: jp.ndarray) -> State:
    """Runs one timestep of the environment's dynamics."""
    data0 = state.pipeline_state
    data = self.pipeline_step(data0, action)

    angle_error = state.info['target_angle'][0] - data.qpos[0]
    # Smooth fall-off on angle reward. Exp is too costly normally,
    # should replace it later on.
    angle_reward = jp.exp(-self._angle_reward_weight*angle_error*angle_error)
    ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

    obs = self._get_obs(data, action, state.info)
    reward = angle_reward - ctrl_cost
    done = 0.0
    state.metrics.update(
        angle_reward=angle_reward,
        reward_quadctrl=-ctrl_cost,
    )

    return state.replace(
        pipeline_state=data, obs=obs, reward=reward, done=done
    )

  def _get_obs(
      self, data: mjx.Data, action: jp.ndarray, info
  ) -> jp.ndarray:
    """Observes elbow angle, velocities, and last applied torque."""
    position = data.qpos

    # external_contact_forces are excluded
    return jp.concatenate([
        position,
        data.qvel,
        data.qfrc_actuator,
        info['target_angle']
    ])


def main(is_msk=True):

    # https://github.com/google-deepmind/mujoco/blob/2546fcefa0e850265028a67ea1d88a83ff14cc10/src/engine/engine_util_misc.c#L458
    # normalized muscle length-gain curve
    def mju_muscleGainLength(length, lmin, lmax, mjMINVAL=1e-15):

        def true_fn(length, lmin, lmax):

            # mid-ranges (maximum is at 1.0)
            a = 0.5*(lmin+1)
            b = 0.5*(1+lmax)

            index = jp.where(length <= a, 0,
                    jp.where(length <= 1, 1,
                    jp.where(length <= b, 2, 3)))

            def branch_0(a, b, length, lmin, lmax):
                x = (length-lmin) / jp.maximum(mjMINVAL, a-lmin)
                return 0.5*x*x
            def branch_1(a, b, length, lmin, lmax):
                x = (1-length) / jp.maximum(mjMINVAL, 1-a)
                return 1 - 0.5*x*x
            def branch_2(a, b, length, lmin, lmax):
                x = (length-1) / jp.maximum(mjMINVAL, b-1)
                return 1 - 0.5*x*x
            def branch_3(a, b, length, lmin, lmax):
                x = (lmax-length) / jp.maximum(mjMINVAL, lmax-b)
                return 0.5*x*x

            return jax.lax.switch(index, [branch_0, branch_1, branch_2, branch_3], a, b, length, lmin, lmax)

        def false_fn(length, lmin, lmax):
            return 0.

        return jax.lax.cond(jp.logical_and(lmin <= length, length <= lmax), true_fn, false_fn, length, lmin, lmax) 

    # https://github.com/google-deepmind/mujoco/blob/2546fcefa0e850265028a67ea1d88a83ff14cc10/src/engine/engine_util_misc.c#L485
    # muscle active force, prm = (range[2], force, scale, lmin, lmax, vmax, fpmax, fvmax)
    def mju_muscleGain(length, vel, lengthrange, acc0, prm, mjMINVAL=1e-15):

        #  unpack parameters
        range    = [prm[0], prm[1]]
        force    = prm[2]
        scale    = prm[3]
        lmin     = prm[4]
        lmax     = prm[5]
        vmax     = prm[6]
        fvmax    = prm[8]

        # scale force if negative, F0        
        force = jax.lax.cond(force < 0, lambda scale, acc0, force: scale / jp.maximum(mjMINVAL, acc0), lambda scale, acc0, force: force, scale, acc0, force) 

        # optimum resting length
        L0 = (lengthrange[1]-lengthrange[0]) / jp.maximum(mjMINVAL, range[1]-range[0])

        # normalized length and velocity
        L = range[0] + (length-lengthrange[0]) / jp.maximum(mjMINVAL, L0)
        V = vel / jp.maximum(mjMINVAL, L0*vmax)

        # length curve
        FL = mju_muscleGainLength(L, lmin, lmax)

        # velocity curve
        y = fvmax-1
    
        index = jp.where(V <= -1, 0,
                jp.where(V <= 0, 1,
                jp.where(V <= y, 2, 3)))

        def branch_0(y, V, fvmax):
            return 0.
        def branch_1(y, V, fvmax):
            return (V+1)*(V+1)
        def branch_2(y, V, fvmax):
            return fvmax - (y-V)*(y-V) / jp.maximum(mjMINVAL, y)
        def branch_3(y, V, fvmax):
            return fvmax

        FV = jax.lax.switch(index, [branch_0, branch_1, branch_2, branch_3], y, V, fvmax)

        # compute FVL and scale, make it negative
        return -force*FL*FV
    
    batch_mju_muscleGain = jax.vmap(mju_muscleGain)

    # https://github.com/google-deepmind/mujoco/blob/2546fcefa0e850265028a67ea1d88a83ff14cc10/src/engine/engine_util_misc.c#L531
    # muscle passive force, prm = (range[2], force, scale, lmin, lmax, vmax, fpmax, fvmax)
    def mju_muscleBias(length, lengthrange, acc0, prm, mjMINVAL=1e-15):
    
        # unpack parameters
        range    = [prm[0], prm[1]]
        force    = prm[2]
        scale    = prm[3]
        lmax     = prm[5]
        fpmax    = prm[7]

        # scale force if negative
        force = jax.lax.cond(force < 0, lambda scale, acc0, force: scale / jp.maximum(mjMINVAL, acc0), lambda scale, acc0, force: force, scale, acc0, force) 

        # optimum length
        L0 = (lengthrange[1]-lengthrange[0]) / jp.maximum(mjMINVAL, range[1]-range[0])

        # normalized length
        L = range[0] + (length-lengthrange[0]) / jp.maximum(mjMINVAL, L0)

        # half-quadratic to (L0+lmax)/2, linear beyond
        b = 0.5*(1+lmax)

        index = jp.where(L <= 1, 0,
                jp.where(L <= b, 1, 2))

        def branch_0(L, b, force, fpmax):
            return 0.
        def branch_1(L, b, force, fpmax):
            x = (L-1) / jp.maximum(mjMINVAL, b-1)
            return -force*fpmax*0.5*x*x
        def branch_2(L, b, force, fpmax):
            x = (L-b) / jp.maximum(mjMINVAL, b-1)
            return -force*fpmax*(0.5 + x)

        return jax.lax.switch(index, [branch_0, branch_1, branch_2], L, b, force, fpmax)
    
    batch_mju_muscleBias = jax.vmap(mju_muscleBias)

    envs.register_environment('elbow', Elbow)

    """## Train Elbow Policy
    
    Let's now train a policy with PPO to move the elbow to a target angle. Training takes about 9-10 minutes on a Tesla A100 GPU.
    """

    print("Building environment")

    env_name = 'elbow'
    backend = 'positional' # @param ['generalized', 'positional', 'spring']
    env = envs.create(env_name=env_name, backend=backend, qpos0=jp.pi*1/4)

    def get_v_dot(actions):
         
        state = env.reset(rng=jax.random.PRNGKey(0))

        next_state = env.step(state, actions)

        tau = next_state.pipeline_state.qfrc_passive + next_state.pipeline_state.qfrc_actuator + next_state.pipeline_state.qfrc_applied
        JTf = next_state.pipeline_state.efc_J.T @ next_state.pipeline_state.efc_force
        c = next_state.pipeline_state.qfrc_bias
        M_inv = jp.linalg.inv(next_state.pipeline_state.qM)
        v_dot = M_inv @ (tau + JTf - c) # v_dot = M^-1 * (tau + J.T @ f - c)

        # gain_force = batch_mju_muscleGain(state.pipeline_state.actuator_length,
        #                     state.pipeline_state.actuator_velocity,
        #                     env.sys.mj_model.actuator_lengthrange,
        #                     env.sys.mj_model.actuator_acc0,
        #                     env.sys.mj_model.actuator_gainprm)

        # bias_force = batch_mju_muscleBias(state.pipeline_state.actuator_length,
        #                         env.sys.mj_model.actuator_lengthrange,
        #                         env.sys.mj_model.actuator_acc0,
        #                         env.sys.mj_model.actuator_biasprm)
        
        # gain_force * state.pipeline_state.act + bias_force
        # state.pipeline_state.actuator_force

        return v_dot
    
    batch_get_v_dot = jax.vmap(get_v_dot, in_axes=(1,))

    def entropy(sigma_squared):

        ent = 0.5 * jp.log(2 * jp.pi * jp.exp(1) * sigma_squared)

        return ent

    def empowerment(a, theta, acc_std, power_var):

        cos_theta, sin_theta = jp.cos(theta), jp.sin(theta)
        B = jp.array(((cos_theta, -sin_theta), (sin_theta, cos_theta)))
        synergy = B @ jp.array([1, 0])

        # p(s’|s,z) = N(s + theta z, sigma^2I)
        # p(y’|s,z) = N(c.T s + c.T theta z, c.T sigma^2I c)
        # p(z) = N(0, P)
        # p(y’|s) = N(c.T s, c.T theta P theta.T c + c.T sigma^2I c)
        # H = 0.5 * log(2 pi e sigma**2)
        # H[Y] - H[Y|Z] = 0.5 * log(2 pi e c.T theta P theta.T c + c.T sigma^2I c) - 0.5 * log(2 pi e c.T sigma^2I c)

        H_y_prime = entropy(jp.dot(a, synergy) * power_var * jp.dot(a, synergy) + acc_std**2)
        H_y_prime_given_z = entropy(acc_std**2)

        emp = H_y_prime - H_y_prime_given_z

        return emp, synergy

    batch_empowerment = jax.vmap(empowerment, in_axes=(None, 0, None, None))

    # def p_of_a():

    #     p(a|theta, z)p(z)

    #     theta z

    #     return 

    N = 100
    x = jp.linspace(0, 1, N)
    a1, a2 = jp.meshgrid(x, x)
    # actions = 0.1 + 0.01 * jax.random.normal(jax.random.PRNGKey(0), (1000, 2))
    actions = np.vstack((a1.flatten(), a2.flatten()))
    v_dot = batch_get_v_dot(actions) * env.dt
    acc_std = 0.1
    noisy_v_dot = v_dot.squeeze() + 0.1 * jax.random.normal(jax.random.PRNGKey(0), (N**2))

    # https://www.cs.toronto.edu/~rgrosse/courses/csc411_f18/slides/lec19-slides.pdf
    X = jp.vstack((actions, jp.ones(N**2))).T
    beta_MLE = jp.linalg.inv(X.T@X) @ X.T @ v_dot

    prior_var = .2
    Sigma_post = jp.linalg.inv(X.T@X/acc_std**2 + jp.eye(3)/prior_var)
    mu_post = Sigma_post/acc_std**2 @ X.T @ v_dot

    power_var = 0.3**2
    thetas = jp.linspace(-jp.pi,jp.pi,100)
    emp, synergies = batch_empowerment(beta_MLE[:2,0], thetas, acc_std, power_var)

    import distrax
    dist_a = distrax.MultivariateNormalFullCovariance(loc=mu_post[:2,0][None].repeat(10_000, axis=0), covariance_matrix=Sigma_post[:2,:2][None].repeat(10_000, axis=0))
    post_a_samp = dist_a.sample(seed=jax.random.PRNGKey(0))
    post_a_samp /= jp.linalg.norm(post_a_samp, axis=-1, keepdims=True)
    post_cov = jax.vmap(lambda x: jp.outer(x,x))(post_a_samp).mean(axis=0)

    dist_a = distrax.MultivariateNormalFullCovariance(loc=jp.zeros((10_000,2)), covariance_matrix=(jp.eye(2)*prior_var)[None].repeat(10_000, axis=0))
    prior_a_samp = dist_a.sample(seed=jax.random.PRNGKey(0))
    prior_a_samp /= jp.linalg.norm(prior_a_samp, axis=-1, keepdims=True)
    prior_cov = jax.vmap(lambda x: jp.outer(x,x))(prior_a_samp).mean(axis=0) * power_var
    
    def evaluate_density_of_source_on_grid_of_actions(cov, a_min=0, a_max=1, N=150):
    
        # create grid of actions
        x_grid = jp.linspace(a_min, a_max, N)
        y_grid = jp.linspace(a_min, a_max, N)
        xv, yv = jp.meshgrid(x_grid, y_grid)
        action = jp.concatenate((xv[:,:,None], yv[:,:,None]),axis=2).reshape((-1,2))

        dist_u = distrax.MultivariateNormalFullCovariance(loc=jp.ones(2,)*0.5, covariance_matrix=cov)

        l_p = dist_u.log_prob(action)

        # convert log prob to prob
        prob = jp.exp(l_p).reshape((N,N))
        
        # flip the vertical axis as grid points start from the top
        prob = jp.flipud(prob.reshape((N,N)))
    
        return prob

    grid_max = 1
    grid_min = 0
    grid_size = 150
    prior_prob = evaluate_density_of_source_on_grid_of_actions(prior_cov, grid_min, grid_max, grid_size)
    post_prob = evaluate_density_of_source_on_grid_of_actions(post_cov, grid_min, grid_max, grid_size)

    from matplotlib import pyplot as plt
    f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(13, 2))
    colormap = plt.cm.get_cmap('RdYlBu')
    sc = ax1.scatter(actions[0,:], actions[1,:], c=v_dot, s=10, cmap=colormap)
    cb = plt.colorbar(sc, label=r"$\ddot{q}$", ticks=[-1,1]) # label=r"$\ddot{q}$"
    cb.set_label(label=r"$\ddot{q}$",labelpad=-15)
    # cb.ax.xaxis.set_label_coords(0, 1)
    ax1.axis('equal')
    ax1.axis('square')
    # sc = ax1.imshow(np.flipud(v_dot.reshape(100,100)), cmap=colormap)
    # cb = plt.colorbar(sc, label="elbow joint acceleration", ticks=[-1,0,1]) # label=r"$\ddot{q}$"
    ax1.set_xlabel('triceps command') # 'triceps long\nmuscle command
    ax1.set_ylabel('biceps command')
    ax1.yaxis.set_label_coords(-0.15, 0.5)
    ax1.xaxis.set_label_coords(+0.5, -.13)
    ax1.set_xticks(ticks=[0, 1])
    ax1.set_yticks(ticks=[1, 0])
    # ax1.set_xticks(ticks=[0.1, 99.5], labels = [0, 1])
    # ax1.set_yticks(ticks=[0.1, 99.5], labels = [1, 0])
    ax1.set_title('dynamics', fontsize=10, y=1.04)
    ax1.spines[['right', 'top']].set_visible(False)
    ax1.spines['bottom'].set_position(('data', 0)) # X-axis at y=0
    ax1.spines['left'].set_position(('data', 0)) # Y-axis at x=0
    # Set the axes limits to the upper-right quadrant
    ax1.set_xlim(0, 1) # only x > 0
    ax1.set_ylim(0, 1) # only y > 0
    sc.set_clip_on(True) # clip points outside the axes limits
    # ax1.arrow(0.5,0.5,beta_MLE[0],beta_MLE[1], head_width=0.3, color='k')
    # ax1.arrow(0.5,0.5,-line_length*2,line_length*2, head_width=0.3, color='k')
    # ax1.text(0.8, 0.72, r'$\text{low}$''\n''$\mathcal{E}$', rotation = 45, horizontalalignment='center')
    # ax1.text(0.2, 0.66, r'$\text{high}$''\n''$\mathcal{E}$', rotation = -45, horizontalalignment='center')
        
    # sc2 = ax2.scatter(synergies[:,0], synergies[:,1], c=emp, s=10, cmap='viridis')
    # cb2 = plt.colorbar(sc2, label=r"$\mathcal{E}$", ticks=[0.3,1.7])
    # cb2.set_label(labelpad=-100)
    # ax2.axis('equal')
    # ax2.axis('square')
    # ax2.set_xticks(ticks=[])
    # ax2.set_yticks(ticks=[])
    # ax2.spines[['right', 'top']].set_visible(False)
    # ax2.yaxis.set_label_coords(-0.17, 0.5)
    # ax2.xaxis.set_label_coords(+0.5, -.13)
    # ax2.set_title('empowerment', fontsize=10, y=1.04)

    # Create the second subplot as a polar plot
    ax2.spines[['right', 'top', 'bottom', 'left']].set_visible(False)
    ax2.set_xticks(ticks=[])
    ax2.set_yticks(ticks=[])
    ax2 = plt.subplot(152, projection='polar')  # Specify polar projection
    ax2.set_xlabel(r'synergy angle $(\theta)$')
    ax2.xaxis.set_label_coords(+0.5, -.13)
    # degrees = [0, 180]
    # radians = np.radians(degrees)  # Convert degrees to radians
    # ax2.set_xticks(radians)  # Set the angular ticks
    # ax2.set_xticklabels(degrees)  # Set the labels as degrees
    ax2.set_xticks([])  # Set the angular ticks
    sc2 = ax2.scatter(thetas, jp.ones(thetas.shape), c=emp, s=25, cmap='magma')
    cb2 = plt.colorbar(sc2, label=r"$\mathcal{E}$", ticks=[0.3,1.7])
    cb2.set_label(label=r"$\mathcal{E}$",labelpad=-15)
    ax2.set_rticks([])  # Remove radial ticks and their labels
    ax2.set_title('empowerment', fontsize=10, y=1.04)
    ax2.grid(False) # Disable gridlines
    ax2.spines['polar'].set_visible(False)  # Remove outer circle
    ax2.set_rlim(0,1.1)
    # ax2.text(thetas[0], 1.1, r'{}$^\circ$'.format(int(jp.degrees(thetas[0]))), fontsize=10, ha='center', va='center')
    ax2.text(1.5*jp.pi, 1.25, r'270$^\circ$', fontsize=10, ha='center', va='center')
    # ax2.text(.5*jp.pi, 1.15, r'90$^\circ$', fontsize=10, ha='center', va='center')
    ax2.text(jp.pi/4, 1.25, r'45$^\circ$', fontsize=10, ha='center', va='center')
    # ax2.text(jp.pi/4*3, 1.1, r'135$^\circ$', fontsize=10, ha='center', va='center')
    # ax2.text(-jp.pi/4, 1.1, r'315$^\circ$', fontsize=10, ha='center', va='center')

    # sc3 = ax3.scatter(synergies[:,0], synergies[:,1], c=emp, s=10, cmap='viridis')
    # cb3 = plt.colorbar(sc3, label=r"$\mathcal{E}$", ticks=[0.3,1.7])
    # cb3.set_label(label=r"$\mathcal{E}$",labelpad=-15)
    # ax3.set_xticks(ticks=[])
    # ax3.set_yticks(ticks=[])
    # ax3.axis('equal')
    # ax3.axis('square')
    # ax3.spines[['right', 'top', 'bottom', 'left']].set_visible(False)
    # ax3.set_title('empowerment', fontsize=10, y=1.04)

    angle_rad = jax.vmap(lambda a_samp: jp.arctan2(a_samp[1], a_samp[0]))(prior_a_samp)
    # angle_deg = jp.degrees(angle_rad)
    count, bin_edges = jp.histogram(angle_rad, bins=thetas, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax3.spines[['right', 'top', 'bottom', 'left']].set_visible(False)
    ax3.set_xticks(ticks=[])
    ax3.set_yticks(ticks=[])
    ax3 = plt.subplot(153, projection='polar')  # Specify polar projection
    ax3.set_xlabel(r'synergy angle $(\theta)$')
    ax3.xaxis.set_label_coords(+0.5, -.13)
    ax3.set_xticks([])  # Set the angular ticks
    sort_idx = np.argsort(count)
    sc3 = ax3.scatter(bin_centers[sort_idx], jp.ones(bin_centers.shape), c=count[sort_idx].mean()*jp.ones(bin_centers.shape), s=25, cmap='Blues', vmin=0)
    # sc3 = ax3.scatter(bin_centers[sort_idx], jp.ones(bin_centers.shape), c=jp.ones(bin_centers.shape), s=10, cmap='Reds')
    cb3 = plt.colorbar(sc3, label=r"$p(\theta)$", ticks=[0,0.15])
    cb3.set_label(label=r"$p(\theta)$",labelpad=-20)
    cb3.set_ticklabels(['0','0.15'])
    ax3.set_rticks([])  # Remove radial ticks and their labels
    ax3.set_title('synergy prior', fontsize=10, y=1.04)
    ax3.grid(False) # Disable gridlines
    ax3.spines['polar'].set_visible(False)  # Remove outer circle
    ax3.set_rlim(0,1.1)
    ax3.text(1.5*jp.pi, 1.25, r'270$^\circ$', fontsize=10, ha='center', va='center')
    ax3.text(jp.pi/4, 1.25, r'45$^\circ$', fontsize=10, ha='center', va='center')

    post_angle_rad = jax.vmap(lambda a_samp: jp.arctan2(a_samp[1], a_samp[0]))(post_a_samp)
    # angle_deg = jp.degrees(angle_rad)
    post_count, post_bin_edges = jp.histogram(post_angle_rad, bins=thetas, density=True)
    post_bin_centers = (post_bin_edges[:-1] + post_bin_edges[1:]) / 2
    ax4.spines[['right', 'top', 'bottom', 'left']].set_visible(False)
    ax4.set_xticks(ticks=[])
    ax4.set_yticks(ticks=[])
    ax4 = plt.subplot(154, projection='polar')  # Specify polar projection
    ax4.set_xlabel(r'synergy angle $(\theta)$')
    ax4.xaxis.set_label_coords(+0.5, -.13)
    ax4.set_xticks([])  # Set the angular ticks
    post_sort_idx = np.argsort(post_count)
    sc4 = ax4.scatter(post_bin_centers[post_sort_idx], jp.ones(post_bin_centers.shape), c=post_count[post_sort_idx], s=25, cmap='Blues')
    cb4 = plt.colorbar(sc4, label=r"$p(\theta|\mathcal{D})$", ticks=[0, 15])
    cb4.set_ticklabels(['0','15'])
    cb4.set_label(label=r"$p(\theta|\mathcal{D})$",labelpad=-10)
    ax4.set_rticks([])  # Remove radial ticks and their labels
    ax4.set_title('synergy posterior', fontsize=10, y=1.04)
    ax4.grid(False) # Disable gridlines
    ax4.spines['polar'].set_visible(False)  # Remove outer circle
    ax4.set_rlim(0,1.1)
    ax4.text(1.5*jp.pi, 1.25, r'270$^\circ$', fontsize=10, ha='center', va='center')
    ax4.text(jp.pi/4, 1.25, r'45$^\circ$', fontsize=10, ha='center', va='center')
    
    ax5.set_xticks(ticks=[])
    ax5.set_yticks(ticks=[])
    ax5.set_title('p. ratio, det(Sigma), emp', fontsize=10, y=1.04)

    plt.savefig('v_dot.png', bbox_inches="tight") 

    # ax3.imshow(prior_prob, cmap='inferno')
    # v = (grid_max-1)/(grid_max-grid_min)*grid_size
    # ax3.set_xticks(ticks=[v,grid_size-v],labels=[0, 1])
    # ax3.set_yticks(ticks=[v,grid_size-v],labels=[1, 0])
    # ax3.set_title('p(a)', fontsize=10, y=1.04)

    breakpoint()

    sc = ax3.imshow(prob, cmap='inferno')
    ax3.set_xlabel('triceps command')
    ax3.set_ylabel('biceps command')
    v = (grid_max-4)/(grid_max-grid_min)*grid_size
    ax3.set_xticks(ticks=[v,grid_size-v],labels=[-4,4])
    ax3.set_yticks(ticks=[v,grid_size-v],labels=[4,-4])
    ax3.set_title('initial p(a)',fontsize=10,y=1.04)
    box = ax3.get_position()
    box.x0 = box.x0 + 0.01
    box.x1 = box.x1 + 0.01
    ax3.set_position(box)
    ax3.yaxis.set_label_coords(-0.12, 0.5)
    ax3.xaxis.set_label_coords(+0.5, -.13)

    plt.savefig('v_dot.png', bbox_inches="tight") 

    # breakpoint()
    
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)

    rollouts = []
    for episode in range(20):
        rng = jax.random.PRNGKey(seed=episode)
        state = jit_env_reset(rng=rng)

        #   <_MjModelCameraViews
        #     bodyid: array([0], dtype=int32)
        #     fovy: array([45.])
        #     id: 0
        #     ipd: array([0.068])
        #     mat0: array([ 0.   ,  0.447, -0.894, -1.   ,  0.   ,  0.   ,  0.   ,  0.894,  0.447])
        #     mode: array([2], dtype=int32)
        #     name: 'side_view'
        #     pos: array([-1.134,  0.033,  1.336])
        #     pos0: array([-1.134,  0.033,  1.336])
        #     poscom0: array([-1.134,  0.033,  1.336])
        #     quat: array([ 0.602,  0.372, -0.372, -0.602])
        #     targetbodyid: array([-1], dtype=int32)
        #     user: array([], dtype=float64)
        #     >

        import mediapy as media
        import os
        cwd = os.path.dirname(os.path.abspath(__file__))
        #   env.sys = env.sys.replace(cam_targetbodyid = mujoco.mj_name2id(env.sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'r_humerus'))
        #   env.sys.mj_model = env.sys.mj_model.replace(cam_pos = np.array([-1.134, 0.033, 1.336]))
        #   env.sys.mj_model.cam_pos0 = np.array([-1.134, 0.033, 1.336])
        #   env.sys.mj_model.camera('front_view').pos
        #   env.sys.mj_model.camera('side_view').pos
        #   camera = env.sys.mj_model.camera('front_view').pos
        # camera.pos = np.array([-1.134, 0.033, 1.336])
        env.sys.mj_model.camera('side_view').pos = np.array([-1.250, 0.071, 0.182])
        env.sys.mj_model.camera('side_view').pos0 = np.array([-1.250, 0.071, 0.182])
        env.sys.mj_model.camera('side_view').poscom0 = np.array([-1.250, 0.071, 0.182])

        from scipy.spatial.transform import Rotation as R

        xyaxes = np.array([[0.010, -1.000, -0.000], [0.091, 0.001, 0.996]])

        # Step 1: Compute the Z-axis using cross product
        Z = np.cross(xyaxes[0], xyaxes[1])

        # Step 2: Construct the full rotation matrix (mat0)
        mat0 = np.vstack([xyaxes, Z]).T  # Stack X, Y, Z into a 3x3 matrix

        # Step 3: Convert the rotation matrix to quaternion
        rotation = R.from_matrix(mat0)
        quat = rotation.as_quat()  

        env.sys.mj_model.camera('side_view').mat0 = mat0.flatten()
        env.sys.mj_model.camera('side_view').quat = quat

        env.sys.mj_model.camera('side_view').fovy = np.array([45.])

        #   media.write_image(cwd + '/image.png', env.render(state.pipeline_state, camera='side_view', height=480, width=640), 'png') 

        breakpoint()

        # https://mujoco.readthedocs.io/en/stable/computation/index.html#general-framework
        # https://mujoco.readthedocs.io/en/stable/modeling.html#muscles
        # https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-muscle
        # https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mju-musclegain
        # https://mujoco.readthedocs.io/en/3.0.0/APIreference/APItypes.html#mjtgain
        #   state.pipeline_state.qpos
        #   state.pipeline_state.qvel
        #   state.pipeline_state.qM # inertia in joint space
        #   state.pipeline_state.qfrc_bias # bias force: Coriolis, centrifugal, gravitational

        #   tau = state.pipeline_state.qfrc_passive # passive forces from spring-dampers and fluid dynamics
        #         + state.pipeline_state.qfrc_actuator # actuation forces, jp.dot(state.pipeline_state.actuator_moment.squeeze(), state.pipeline_state.actuator_force)
        #         + state.pipeline_state.qfrc_applied #  additonal forces specified by the user.

        #     state.pipeline_state.efc_J # constraint Jacobian
        #     state.pipeline_state.efc_force # constraint force

        # env.sys.mj_model.actuator_dynprm.shape # time constants

        # env.sys.mj_model.actuator_lengthrange
        # env.sys.mj_model.actuator_length0
        # env.sys.mj_model.actuator_gainprm
        # env.sys.mj_model.actuator_biasprm # same as above

        tau = state.pipeline_state.qfrc_passive + state.pipeline_state.qfrc_actuator + state.pipeline_state.qfrc_applied
        JTf = state.pipeline_state.efc_J.T@state.pipeline_state.efc_force
        c = state.pipeline_state.qfrc_bias
        M_inv = jp.linalg.inv(state.pipeline_state.qM)
        v_dot = M_inv @ (tau + JTf - c) # v_dot = M^-1 * (tau + J.T @ f - c)

        breakpoint()

        # state.pipeline_state.ctrl 
        #   state.pipeline_state.actuator_force 
        #   state.pipeline_state.actuator_length 
        #   state.pipeline_state.actuator_velocity 
        #   state.pipeline_state.act 
        # state.pipeline_state.act_dot

        rollout = {}
        rollout['IFtip_target'] = state.info['IFtip_target']
        states = []
        while not (state.done or state.info['truncation']):
            states.append(state.pipeline_state)
            act_rng, rng = jax.random.split(rng)
            act, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_env_step(state, act)

    env_name = 'elbow'
    env = envs.get_environment(env_name, is_msk=is_msk)


    def check_env(model, data):
        obs = env._get_obs(data, data.ctrl)
        assert not np.any(np.isnan(obs))
        angle_error = 1 - data.qpos[0]
        angle_reward = np.exp(-env._angle_reward_weight * angle_error * angle_error)
        ctrl_cost = env._ctrl_cost_weight * np.sum(np.square(data.ctrl))
        reward = angle_reward - ctrl_cost
        data.ctrl = np.random.uniform(-1, 1, (env.action_size,))
        assert not np.isnan(reward)

    train_fn = functools.partial(
        ppo.train, num_timesteps=20_000_000, num_evals=5, reward_scaling=0.1,
        episode_length=1000, normalize_observations=True, action_repeat=1,
        unroll_length=10, num_minibatches=1, num_updates_per_batch=8,
        discounting=0.97, learning_rate=3e-4, entropy_cost=1e-3, num_envs=10,
        batch_size=10, seed=0)

    x_data = []
    y_data = []
    ydataerr = []
    times = [datetime.now()]

    max_y, min_y = 5000, 0
    # Plot learning curves
    def progress(num_steps, metrics):
      times.append(datetime.now())
      x_data.append(num_steps)
      y_data.append(metrics['eval/episode_reward'])
      ydataerr.append(metrics['eval/episode_reward_std'])

      plt.xlim([0, train_fn.keywords['num_timesteps'] * 1.25])
      plt.ylim([min_y, max_y])

      plt.xlabel('# environment steps')
      plt.ylabel('reward per episode')
      plt.title(f'y={y_data[-1]:.3f}')

      plt.errorbar(
          x_data, y_data, yerr=ydataerr)
      plt.savefig(f'{num_steps}.png')


    # Instantiate the environment then train
    print("Jitting then training")
    make_inference_fn, params, _= train_fn(environment=env, progress_fn=progress)

    print(f'time to jit: {times[1] - times[0]}')
    print(f'time to train: {times[-1] - times[1]}')


    #Save Model
    model_path = './elbow_params.pickle'
    model.save_params(model_path, params)

if __name__ == '__main__':
    main()