# -*- coding: utf-8 -*-
from gym.envs.classic_control.cartpole import CartPoleEnv
import math

import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

# create custom environment inheriting the original cartpole environment
class CartPoleDR(CartPoleEnv):
    def __init__(self, dr_params):
        super().__init__()
        self.Nc_pre = 1
        self.rand_force_mu = 10.0
        self.rand_force_sigma = dr_params['force_sigma']
        self.rand_friction_cart_hi = dr_params['friction_cart_hi']
        self.rand_friction_pole_hi = dr_params['friction_pole_hi']
        self.obs_noise_x = dr_params['observe_x_sigma']
        self.obs_noise_x_dot = dr_params['observe_xdot_sigma']
        self.obs_noise_theta = dr_params['observe_theta_sigma']
        self.obs_noise_theta_dot = dr_params['observe_thetadot_sigma']
        self.step_count = 0
        self.train = True
        self.randomize_friction()

    def set_force_rand_param(self, force_mu: float, force_sigma: float):
        """ ステップ毎に force を変える際の正規分布パラメータ """
        self.rand_force_mu = force_mu
        self.rand_force_sigma = force_sigma

    def set_friction_rand_param(self, friction_cart_hi: float, friction_pole_hi: float):
        """ ステップ毎に force を変える際の正規分布パラメータ """
        self.rand_friction_cart_hi = friction_cart_hi
        self.rand_friction_pole_hi = friction_pole_hi
        
    def set_frictions(self, friction_cart: float, friction_pole: float):
        """ 摩擦係数の設定 """
        self.friction_cart = friction_cart
        self.friction_pole = friction_pole

    def randomize_friction(self):
        ''' 環境にランダム性を与えます '''
        # Friction
        self.friction_cart = np.random.uniform(0.0, self.rand_friction_cart_hi)
        self.friction_pole = np.random.uniform(0.0, self.rand_friction_pole_hi)

        print(f'friction_cart: {self.friction_cart}')        
        
    def get_state(self):
        return np.array(self.state)

    def change_pole_length(self, length=0.5):
        """ Pole の長さを変更する """
        self.length = length

    def step(self, action):
        """ Cartpole の step を overwriteし
            摩擦係数を考慮した物理モデルへ変更する
        """
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        x, x_dot, theta, theta_dot = self.state
        # force = self.force_mag if action == 1 else -self.force_mag
        # Domain Randomization. Randomize force every step
        force = np.random.normal(self.rand_force_mu, self.rand_force_sigma)
        force = force if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf

        """ 下記コメントアウトは摩擦を考慮していない、元の実装 """
        # temp = (
        #     force + self.polemass_length * theta_dot ** 2 * sintheta
        # ) / self.total_mass
        # thetaacc = (self.gravity * sintheta - costheta * temp) / (
        #     self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        # )
        # xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        """ 摩擦を考慮したモデルに変更する。
            詳細は上記 pdf の concluesion を参照。
        """
        thetaacc = self.calc_theta_acc(
            force, theta_dot, x_dot, sintheta, costheta, self.Nc_pre
        )
        Nc = self.calc_Nc(thetaacc, theta_dot, sintheta, costheta)
        if np.sign(Nc) != np.sign(self.Nc_pre):
            thetaacc = self.calc_theta_acc(
                force, theta_dot, x_dot, sintheta, costheta, Nc
            )
        xacc = self.calc_x_acc(
            force, thetaacc, theta_dot, x_dot, sintheta, costheta, Nc
        )
        self.Nc_pre = Nc
        """ 摩擦を考慮したモデルへの変更完了 """

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)
        self.step_count += 1

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
            or self.step_count >= 200
        )

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0
        
        noisy_obs = self.add_noise_on_obs()
        
        # Check
#         print(f'True x: {x}')
#         print(f'Noise x: {noisy_obs[0]}')
        
        return noisy_obs, reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        if (self.train and self.step_count >= 200) or (not self.train):
            # 訓練時200ステップクリアしたら次の摩擦へ。推論時はリセット毎に摩擦を変更する。
            self.randomize_friction()
        self.step_count = 0  # reset the counter
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width/world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.79, .18, .55)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.13, .2, .57)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def mode_train(self):
        ''' 環境を訓練モードにし、成功するまで摩擦を変更しない '''
        print('Mode: Train')
        self.train = True
    
    def mode_eval(self):
        ''' 環境を推論モードにし、リセット毎に摩擦を変更する '''
        print('Mode: Evaluation')
        self.train = False
    
    ###############################################
    # step() 内での計算補助関数
    ###############################################
    def calc_temp(self, F, theta_dot, x_dot, sintheta, costheta, Nc):
        """ support fuction for calc_theta_acc """
        temp = (
            F
            + self.polemass_length
            * (theta_dot ** 2)
            * (sintheta + self.friction_cart * np.sign(Nc * x_dot) * costheta)
        ) / self.total_mass - self.friction_cart * self.gravity * np.sign(Nc * x_dot)

        return temp

    def calc_theta_acc(self, F, theta_dot, x_dot, sintheta, costheta, Nc):
        """ calculate the acceleration of theta"""
        temp = self.calc_temp(F, theta_dot, x_dot, sintheta, costheta, Nc)

        theta_acc = (
            self.gravity * sintheta
            - costheta * temp
            - self.friction_pole * theta_dot / self.polemass_length
        ) / (
            self.length
            * (
                4.0 / 3.0
                - self.masspole
                * costheta
                * (costheta - self.friction_cart * np.sign(Nc * x_dot))
                / self.total_mass
            )
        )

        return theta_acc

    def calc_Nc(self, theta_acc, theta_dot, sintheta, costheta):
        """ calculate Nc """
        Nc = self.total_mass * self.gravity - self.polemass_length * (
            theta_acc * sintheta + theta_dot ** 2 * costheta
        )

        return Nc

    def calc_x_acc(self, F, theta_acc, theta_dot, x_dot, sintheta, costheta, Nc):
        """ calculate the acceleration of position x """
        x_acc = (
            F
            + self.polemass_length * (theta_dot ** 2 * sintheta - theta_acc * costheta)
            - self.friction_cart * Nc * np.sign(Nc * x_dot)
        ) / self.total_mass

        return x_acc

    ###############################################
    # 以下は検算用関数
    # 元の cartpole での計算再現用。
    ###############################################
    def calc_temp_orig(self, force, theta_dot, sintheta):
        """ original cartpole's calculation for temp. For validation. """
        temp = (
            force + self.polemass_length * theta_dot ** 2 * sintheta
        ) / self.total_mass

        return temp

    def calc_theta_acc_orig(self, force, theta_dot, sintheta, costheta):
        """ original cartpole's calculation for acceleration of theta. For validation. """
        temp = self.calc_temp_orig(force, theta_dot, sintheta)
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        return thetaacc

    def calc_x_acc_orig(self, force, theta_acc, theta_dot, sintheta, costheta):
        """ original cartpole's calculation for acceleration of x. For validation. """
        temp = self.calc_temp_orig(force, theta_dot, sintheta)
        xacc = temp - self.polemass_length * theta_acc * costheta / self.total_mass
        return xacc

    def add_noise_on_obs(self):
        x_true, x_dot_true, theta_true, theta_dot_true = self.state
        x_noise = np.random.normal(x_true, self.obs_noise_x)
        x_dot_noise = np.random.normal(x_dot_true, self.obs_noise_x_dot)
        theta_noise = np.random.normal(theta_true, self.obs_noise_theta)
        theta_dot_noise = np.random.normal(theta_dot_true, self.obs_noise_theta_dot)
        
        x_noise = np.clip(x_noise, self.observation_space.low[0], self.observation_space.high[0])
        x_dot_noise = np.clip(x_dot_noise, self.observation_space.low[1], self.observation_space.high[1])
        theta_noise = np.clip(theta_noise, self.observation_space.low[2], self.observation_space.high[2])
        theta_dot_noise = np.clip(theta_dot_noise, self.observation_space.low[3], self.observation_space.high[3])
        
        obs_noise = np.array([x_noise, x_dot_noise, theta_noise, theta_dot_noise])
        return obs_noise

    
def domain_randomize(env, dr_params):
    ''' 環境にランダム性を与えます '''
    # Force
    env.set_force_rand_param(dr_params['force_mu'], dr_params['force_sigma'])
    
    # Frictions
    env.set_friction_rand_param(dr_params['friction_cart_hi'], dr_params['friction_pole_hi'])
    
    return env


def create_env(config):    
    # Domain Randomization (Level1) 
#     dr_params = {
#         'force_mu': 10.0,
#         'force_sigma': 0.3,
#         'friction_cart_lo': 0.0,
#         'friction_cart_hi': 0.3,
#         'friction_pole_lo': 0.0,
#         'friction_pole_hi': 0.3,
#         'observe_x_sigma': 0.24,
#         'observe_xdot_sigma': 0.1,
#         'observe_theta_sigma': 0.01,
#         'observe_thetadot_sigma': 0.01,
#     }

    # Domain Randomization (Level2) dr off でも推論可能
#     dr_params = {
#         'force_mu': 10.0,
#         'force_sigma': 0.8,
#         'friction_cart_lo': 0.0,
#         'friction_cart_hi': 0.75,
#         'friction_pole_lo': 0.0,
#         'friction_pole_hi': 0.75,
#         'observe_x_sigma': 0.24,
#         'observe_xdot_sigma': 0.1,
#         'observe_theta_sigma': 0.01,
#         'observe_thetadot_sigma': 0.01,
#     }    

    # Domain Randomization (Level2.5)
    dr_params = {
        'force_mu': 10.0,
        'force_sigma': 0.8,
        'friction_cart_lo': 0.0,
        'friction_cart_hi': 0.9,
        'friction_pole_lo': 0.0,
        'friction_pole_hi': 0.9,
        'observe_x_sigma': 0.24,
        'observe_xdot_sigma': 0.15,
        'observe_theta_sigma': 0.01,
        'observe_thetadot_sigma': 0.025,
    }


    # Domain Randomization (Level3) dr off の成功率 8/20, dr on 学習 収束せず
#     dr_params = {
#         'force_mu': 10.0,
#         'force_sigma': 1.0,
#         'friction_cart_lo': 0.0,
#         'friction_cart_hi': 1.0,
#         'friction_pole_lo': 0.0,
#         'friction_pole_hi': 1.0,
#         'observe_x_sigma': 0.24,
#         'observe_xdot_sigma': 0.15,
#         'observe_theta_sigma': 0.01,
#         'observe_thetadot_sigma': 0.025,
#     }

    # Domain Randomization (Level4) c
#     dr_params = {
#         'force_mu': 10.0,
#         'force_sigma': 1.2,
#         'friction_cart_lo': 0.0,
#         'friction_cart_hi': 1.5,
#         'friction_pole_lo': 0.0,
#         'friction_pole_hi': 1.5,
#         'observe_x_sigma': 0.24,
#         'observe_xdot_sigma': 0.2,
#         'observe_theta_sigma': 0.1,
#         'observe_thetadot_sigma': 0.05,
#     }        

    # DR Off equivalent
#     dr_params = {
#         'force_mu': 10.0,
#         'force_sigma': 0.0,
#         'friction_cart_lo': 0.0,
#         'friction_cart_hi': 0.0,
#         'friction_pole_lo': 0.0,
#         'friction_pole_hi': 0.0,
#         'observe_x_sigma': 0.0,
#         'observe_xdot_sigma': 0.0,
#         'observe_theta_sigma': 0.0,
#         'observe_thetadot_sigma': 0.0,
#     }    

    env = CartPoleDR(dr_params)
    env = domain_randomize(env, dr_params)
    env.change_pole_length(2.5*0.5)
    
    if config['train'] == True:
        env.mode_train()
    else:
        env.mode_eval()
    
    return env
