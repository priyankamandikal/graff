'''
Env for grasping task
Can also be run for debugging as:
    python grasp.py --obj pan --multicontact threshold --multivalue 0.5 --scale 1
    python grasp.py --obj pan --multicontact percentage --multivalue 0.5 --scale 1
'''

import os
from os.path import join, dirname, abspath, exists

import numpy as np
import quaternion as qt
from pyquaternion import Quaternion
from envs.mj_envs.utils.quat_utils import *
from gym import utils
from envs.mj_envs.utils import mujoco_env
# from gym.envs.mujoco import mujoco_env
from gym.utils import seeding
from mujoco_py import MjViewer, MjSim, functions, load_model_from_path, MjRenderContext
import json
import cv2

class GraffV0(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, object, device_id=0, process_id=0, grasp_attrs_dict=None):

        self.grasp_attrs_dict = grasp_attrs_dict
        self.obj_name = object
        self.rewards = self.grasp_attrs_dict['rewards']
        self.S_grasp_sid = 0
        self.obj_bid = 0
        self.target_pos = [0.0, 0.0, 0.2]
        self.device_id = device_id
        self.process_id = process_id
        curr_dir = dirname(abspath(__file__))
        if grasp_attrs_dict['dataset'] == 'contactdb':
            model_path = join(curr_dir, 'assets/contactdb/env_xmls/%s_vhacd.xml' % self.obj_name)
        frame_skip = 5
        self.seed()

        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = join(dirname(__file__), "assets", model_path)
        if not exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = load_model_from_path(fullpath)
        self.sim = MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}

        if self.grasp_attrs_dict['policy'] in ['cnn-mlp']:
            self.frame_size = (self.grasp_attrs_dict['img_res'], self.grasp_attrs_dict['img_res'])
            self.imagenet_mean = [0.485, 0.456, 0.406]
            self.imagenet_std = [0.229, 0.224, 0.225]
            self.min_depth = {'fixed': 0.9, 'first_person': 0.9, 'egocentric': 0.9, 'egocentric_zoom': 0.7}
            self.max_depth = {'fixed': 1.0, 'first_person': 1.0, 'egocentric': 1.0, 'egocentric_zoom': 1.0}
            if self.grasp_attrs_dict['debug']:
                self.tmp = 0
                os.makedirs('images/%s'%str(self.process_id), exist_ok=True)

        # load contacts
        if grasp_attrs_dict['dataset'] == 'contactdb':
            self.contact_json = join(curr_dir, 'assets/contactdb/contacts/%s.json'%self.obj_name)
            with open(self.contact_json, 'r') as f:
                data = json.load(f)
                geom_indices = data['multicontact']['geom']
                contact_indices = data['multicontact']['vert']

        # multicontact
        self.n_verts_multicontact = 20
        self.fps_geom_indices, self.fps_contact_indices = self.get_fps(geom_indices, contact_indices, self.n_verts_multicontact)

        # geoms
        self.objgeom_names = [name for name in self.sim.model.geom_names if 'Obj_mesh_' in name]
        self.nobjgeoms = len(self.objgeom_names)

        # hand contact points
        self.touch_sensor_names = [name for name in self.sim.model.sensor_names if 'ST_Tch' in name]
        #TODO: change sensor name
        # self.touch_sensor_names_obs = ["ST_Tch_fftip", "ST_Tch_mftip", "ST_Tch_rftip", "ST_Tch_lftip", "ST_Tch_thtip", "ST_Tch_ffproximal", "ST_Tch_mfproximal", "ST_Tch_rfproximal", "ST_Tch_lfproximal"]
        self.touch_sensor_names_obs = ["ST_Tch_ff_distal", "ST_Tch_mf_distal", "ST_Tch_rf_distal", "ST_Tch_th_distal", "ST_Tch_ff_proximal", "ST_Tch_mf_proximal", "ST_Tch_rf_proximal" ]

        # camera
        self.camera_id_first_person = self.model.camera_name2id('first_person')
        self.camera_id_left = self.model.camera_name2id('left')
        self.camera_id_right = self.model.camera_name2id('right')
        self.camera_id_egocentric = self.model.camera_name2id('egocentric')
        self.cameras = self.grasp_attrs_dict['cameras']

        # initiliaze mujoco env
        mujoco_env.MujocoEnv.__init__(self, model_path, frame_skip)
        print('Observation Space: ', self.observation_space)

        # set ofscreen render contexts
        self.mjr_render_context_offscreen = MjRenderContext(self.sim, offscreen=True,
                                                            # device_id=self.device_id,
                                                            opengl_backend='egl')
        self.sim._render_context_offscreen.vopt.flags[0] = 0
        self.sim._render_context_offscreen.vopt.flags[11] = self.sim._render_context_offscreen.vopt.flags[12] = 1

        # change actuator sensitivity
        #TODO: change actuator sensitivity
        # self.sim.model.actuator_gainprm[
        # self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0') + 1, :3] = np.array(
        #     [10, 0, 0])
        self.sim.model.actuator_gainprm[
        self.sim.model.actuator_name2id('A_ffa0'):self.sim.model.actuator_name2id('A_tha3') + 1, :3] = np.array(
            [1, 0, 0])
        # self.sim.model.actuator_biasprm[
        # self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0') + 1, :3] = np.array(
        #     [0, -10, 0])
        self.sim.model.actuator_biasprm[
        self.sim.model.actuator_name2id('A_ffa0'):self.sim.model.actuator_name2id('A_tha3') + 1, :3] = np.array(
            [0, -1, 0])

        # obtain a few env settings
        self.S_grasp_sid = self.sim.model.site_name2id('S_grasp')
        self.obj_bid = self.sim.model.body_name2id('Object')
        utils.EzPickle.__init__(self)
        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0])

        # set mass of the object
        self.sim.model.body_mass[self.obj_bid] = self.grasp_attrs_dict['obj_mass']

        # set gravity of the environment
        self.sim.model.opt.gravity[:] = [0, 0, self.grasp_attrs_dict['gravity']]

        functions.mj_setConst(self.sim.model, self.sim.data)


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def get_fps(self, geom_indices, contact_indices, K):

        def get_verts(geom_indices, contact_indices):
            contact_verts = []
            for geom_mesh_idx, ind in zip(geom_indices, contact_indices):
                vert = self.model.mesh_vert[self.model.mesh_vertadr[geom_mesh_idx] + ind]
                geom_idx = self.model.geom_name2id('Obj_mesh_%d' % geom_mesh_idx)
                geom_pos = self.model.geom_pos[geom_idx]
                geom_quat = self.model.geom_quat[geom_idx]
                vert = quat_rot_vector(qt.as_quat_array(geom_quat), vert) + geom_pos
                contact_verts.append(vert)
            return contact_verts

        def calc_distances(p0, points):
            return ((p0 - points) ** 2).sum(axis=1)

        pts = get_verts(geom_indices, contact_indices)
        indices = []
        farthest_pts = np.zeros((K, 3))
        indices.append(0)
        farthest_pts[0] = pts[0]
        distances = calc_distances(farthest_pts[0], pts)
        for i in range(1, K):
            next_point = np.argmax(distances)
            indices.append(next_point)
            farthest_pts[i] = pts[next_point]
            distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
        farthest_geoms = [geom_indices[ele] for ele in indices]
        farthest_indices = [contact_indices[ele] for ele in indices]
        return farthest_geoms, farthest_indices


    def get_multicontact_verts(self, verbose=False):
        body_xpos = self.data.body_xpos[self.obj_bid]
        body_xquat = self.data.body_xquat[self.obj_bid].ravel()
        body_xquat = qt.as_quat_array(body_xquat)
        contact_verts = []
        for geom_mesh_idx, contact_idx in zip(self.fps_geom_indices, self.fps_contact_indices):
            vert = self.model.mesh_vert[self.model.mesh_vertadr[geom_mesh_idx] + contact_idx]
            geom_idx = self.model.geom_name2id('Obj_mesh_%d' % geom_mesh_idx)
            geom_pos = self.model.geom_pos[geom_idx].ravel()
            geom_quat = qt.as_quat_array(self.model.geom_quat[geom_idx])
            vert = quat_rot_vector(geom_quat, vert) + geom_pos
            vert = quat_rot_vector(body_xquat, vert) + body_xpos
            contact_verts.append(vert)
        if verbose:
            print('multicontact: ', contact_verts)
        return contact_verts


    def get_multihand_pos(self, sensor_names, verbose=False):
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        touch_pos = []
        for sensor_name in sensor_names:
            site_id = self.sim.model.site_name2id(sensor_name[3:])
            site_pos = self.data.site_xpos[site_id].ravel()
            touch_pos.append(site_pos)
        if verbose:
            print('palm pos:', palm_pos)
            print('touch sensors pos: ', touch_pos)
        return touch_pos


    def chamfer_dist(self, set1, set2):
        
        set1to2 = [np.min(np.linalg.norm(pt-set2, axis=1)) for pt in set1]
        set2to1 = [np.min(np.linalg.norm(pt-set1, axis=1)) for pt in set2]
        chamfer_dist = (np.mean(set1to2) + np.mean(set2to1)) / 2.
        return chamfer_dist


    def step(self, a):
        if self.grasp_attrs_dict['noise']:  # actuation noise
            a = self._gaussian_noise(a, mean=0, std=0.01)
        a = np.clip(a, -1.0, 1.0)
        try:
            a = self.act_mid + a * self.act_rng  # mean center and scale
        except:
            a = a  # only for the initialization phase

        self.do_simulation(a, self.frame_skip)
        ob = self.get_obs()
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()

        rewards = {}
        for reward in ['grasp', 'com', 'aff']:
            rewards[reward] = 0.0

        # distance reward
        if 'com' in self.rewards.keys():
            hand_obj_dist = np.linalg.norm(palm_pos - obj_pos)
            if hand_obj_dist > self.grasp_attrs_dict['reward_dst_thr']:
                rewards['com'] = -1 * self.rewards['com'] * hand_obj_dist  # take hand close to object
        elif 'aff' in self.rewards.keys():
            contact_verts = self.get_multicontact_verts()
            hand_multi_pos = self.get_multihand_pos(self.touch_sensor_names)
            hand_obj_dist = self.chamfer_dist(contact_verts, hand_multi_pos)
            if hand_obj_dist > self.grasp_attrs_dict['reward_dst_thr']:
                rewards['aff'] = -1 * self.rewards['aff'] * hand_obj_dist  # take hand close to object

        # grasp success reward
        # check if obj and hand are in contact with each other but not in contact with table
        # note that contact data buffer has contacts left over from previous iterations, so only check first ncon values
        obj_hand_contact, obj_table_contact, hand_table_contact = False, False, False
        for contact in self.data.contact[:self.data.ncon]:
            contact_pair = sorted(
                [self.model.geom_id2name(contact.geom1), self.model.geom_id2name(contact.geom2)])
            if 'C_' in contact_pair[0] and 'Obj_mesh' in contact_pair[1]:
                obj_hand_contact = True
            if 'Obj_mesh' in contact_pair[0] and 'table' in contact_pair[1]:
                obj_table_contact = True
            if 'C_' in contact_pair[0] and 'table' in contact_pair[1]:
                hand_table_contact = True
        obj_lift = obj_hand_contact and not obj_table_contact and not hand_table_contact
        obj_grab = obj_hand_contact and not obj_table_contact
        if 'grasp' in self.rewards:
            rewards['grasp'] = self.rewards['grasp']*int(obj_grab)
        
        # total reward
        reward_tot = sum(rewards.values())
        info = dict(obj_lift=obj_lift,
                    obj_grab=obj_grab,
                    reward=reward_tot)
        return ob, reward_tot, False, info


    def _get_cnn_obs(self, cam):
        # print('cam: ', cam)
        # print('device_id: ', self.device_id)
        rgbd_frame = self.sim.render(width=self.frame_size[0], height=self.frame_size[1],
                                     mode='offscreen', camera_name=cam,
                                     depth=True,
                                    #  device_id=self.device_id
                                     )
        cnn_inp = {}
        # rgb image
        if 'rgb' in self.grasp_attrs_dict['inputs']:
            cnn_inp['rgb'] = rgbd_frame[0][::-1]  # rgb: (H,W,3) range: [0, 255]
            if self.grasp_attrs_dict['noise']: # img noise
                cnn_inp['rgb'] = cnn_inp['rgb'] + np.random.randint(low=-5, high=6, size=cnn_inp['rgb'].shape)
                cnn_inp['rgb'] = np.clip(cnn_inp['rgb'], 0, 255)
        # depth map
        if 'depth' in self.grasp_attrs_dict['inputs']:
            cnn_inp['depth'] = ((rgbd_frame[1][::-1] - self.min_depth[cam]) / (self.max_depth[cam] - self.min_depth[cam]) * 255).astype(np.uint8)  # (H,W)
            cnn_inp['depth'] = np.expand_dims(cnn_inp['depth'], axis=-1)  # (H,W,1)
        # affordance map
        if 'aff' in self.grasp_attrs_dict['inputs'] and self.sim._render_context_offscreen:
            contact_verts = self.get_multicontact_verts()
            for vert in contact_verts:
                self.sim._render_context_offscreen.add_marker(pos=vert, rgba=[0, 1, 0, 1],
                                                                size=[0.01, 0.01, 0.01],
                                                                label="")  # , type=const.GEOM_SPHERE)
            aff_frame_rgb = self.sim.render(width=self.frame_size[0], height=self.frame_size[1],
                                            mode='offscreen', camera_name=cam,
                                            depth=False, device_id=self.device_id)
            del self.sim._render_context_offscreen._markers[:]
            aff_frame_hsv = cv2.cvtColor(aff_frame_rgb, cv2.COLOR_RGB2HSV)
            cnn_inp['aff'] = cv2.inRange(aff_frame_hsv, (60, 120, 120), (90, 255, 255))[::-1]
            cnn_inp['aff'] = cv2.distanceTransform(cnn_inp['aff'], cv2.DIST_L2, 3)
            cv2.normalize(cnn_inp['aff'], cnn_inp['aff'], 0, 255, cv2.NORM_MINMAX)
            cnn_inp['aff'] = np.expand_dims(cnn_inp['aff'], axis=-1)  # (H,W,1)
        # concatenate inputs
        # Note: python3.6 onward dict is ordered by default
        cnn_inp_concat = np.concatenate([item for key,item in cnn_inp.items()], axis=-1)
        cnn_inp_concat = cnn_inp_concat / 255.
        # debug
        if self.grasp_attrs_dict['debug']:
            self.tmp += 1
            print(self.tmp, cam, rgbd_frame[1].min(), rgbd_frame[1].max())
            print('flags: ', self.sim._render_context_offscreen.vopt.flags)
            print('img: ', cnn_inp['rgb'].dtype, cnn_inp['rgb'].shape, cnn_inp['rgb'].min(), cnn_inp['rgb'].max())
            print('orig depth: ', rgbd_frame[1].dtype, rgbd_frame[1].shape, rgbd_frame[1].min(), rgbd_frame[1].max())
            print('depth: ', cnn_inp['depth'].dtype, cnn_inp['depth'].shape, cnn_inp['depth'].min(), cnn_inp['depth'].max())
            img_frame = cv2.cvtColor(cnn_inp['rgb'].astype(np.float32), cv2.COLOR_BGR2RGB)
            os.makedirs('images/%s/%s' % (str(self.process_id), cam), exist_ok=True)
            cv2.imwrite('images/%s/%s/%d_rgb.png' % (str(self.process_id), cam, self.tmp), img_frame)
            if 'depth' in self.grasp_attrs_dict['inputs']:
                cv2.imwrite('images/%s/%s/%d_depth.png' % (str(self.process_id), cam, self.tmp), cnn_inp['depth'])
            if 'aff' in self.grasp_attrs_dict['inputs']:
                try:
                    cv2.imwrite('images/%s/%s/%d_aff.png' % (str(self.process_id), cam, self.tmp), cnn_inp['aff'])
                except Exception as e:
                    pass
        return cnn_inp_concat


    def _get_mlp_obs(self):
        mlp_inp = {}
        # agent proprioception
        if 'proprio' in self.grasp_attrs_dict['inputs']:
            qp = self.data.qpos.ravel()
            qv = np.clip(self.data.qvel.ravel(), -1.0, 1.0)
            if self.grasp_attrs_dict['noise']:  # proprioceptive noise
                qp = self._gaussian_noise(qp, mean=0, std=0.01)
                qv = self._gaussian_noise(qv, mean=0, std=0.01)
            mlp_inp['proprio'] = np.concatenate([qp[:-6], qv[:-6]]) # [30, 30] qp is in rad
        # object location
        if 'loc' in self.grasp_attrs_dict['inputs']:
            if 'aff' in self.grasp_attrs_dict['inputs']:
                contact_verts = self.get_multicontact_verts() # self.n_verts_multicontact
                hand_multi_pos = self.get_multihand_pos(self.touch_sensor_names_obs) # 9 locations
                if self.grasp_attrs_dict['noise']:
                    contact_verts = self._gaussian_noise(np.array(contact_verts), mean=0, std=0.01)
                    hand_multi_pos = self._gaussian_noise(np.array(hand_multi_pos), mean=0, std=0.01)
                hand_array = np.repeat(hand_multi_pos, self.n_verts_multicontact, axis=0)
                multicontact_array = np.tile(contact_verts, (len(hand_multi_pos), 1))
                dist = (hand_array - multicontact_array).ravel() # self.n_verts_multicontact * 9 * 3
            else:
                palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
                obj_pos = self.data.body_xpos[self.obj_bid].ravel()
                if self.grasp_attrs_dict['noise']:
                    palm_pos = self._gaussian_noise(palm_pos, mean=0, std=0.01)
                    obj_pos = self._gaussian_noise(obj_pos, mean=0, std=0.01)
                dist = palm_pos - obj_pos
            mlp_inp['loc'] = dist
        mlp_inp_concat = np.concatenate([item for key,item in mlp_inp.items()]).astype(np.float32)
        return mlp_inp_concat


    # additive gaussian noise
    def _gaussian_noise(self, var, mean=0, std=0.01):
        noise = np.random.normal(mean, std, var.shape)
        return var+noise


    def get_obs(self):
        obs = {}
        obs['mlp'] = self._get_mlp_obs()
        if self.grasp_attrs_dict['policy'] == 'mlp':
            return obs
        elif self.grasp_attrs_dict['policy'] == 'cnn-mlp':
            cnn_inp = {}
            for cam in self.cameras:
                cnn_inp[cam] = self._get_cnn_obs(cam)
                obs[cam] = cnn_inp[cam].astype(np.float32)
            return obs
        else:
            raise NotImplementedError


    def reset_model(self, angle=None):
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)
        if self.grasp_attrs_dict['obj_tr']:
            # randomly translate object
            self.model.body_pos[self.obj_bid, 0] = self.np_random.uniform(low=-0.15, high=0.15)
            self.model.body_pos[self.obj_bid, 1] = self.np_random.uniform(low=-0.15, high=0.05)
        if self.grasp_attrs_dict['obj_rot']:
            # randomly rotate object, within front semi-circle
            if angle is None:
                if self.obj_name in ['cell_phone', 'stapler', 'teapot', 'toothpaste']:
                    angle = self.np_random.uniform(low=0, high=180)
                else:
                    angle = self.np_random.uniform(low=180, high=360)
            quat = Quaternion(axis=[0,0,1], degrees=angle).elements
            self.model.body_quat[self.obj_bid] = quat
        # forward simulation
        self.sim.forward()
        return self.get_obs()


    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        # hand_qpos = qp[:30]
        hand_qpos = qp[:22]
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        return dict(hand_qpos=hand_qpos, obj_pos=obj_pos, palm_pos=palm_pos,
                    qpos=qp, qvel=qv)


    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        obj_pos = state_dict['obj_pos']
        self.set_state(qp, qv)
        self.model.body_pos[self.obj_bid] = obj_pos
        self.sim.forward()


    def mj_viewer_setup(self, verbose=True):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.fixedcamid = self.camera_id_first_person
        # self.viewer.cam.type = const.CAMERA_FIXED
        self.sim.forward()
        if verbose:
            print('viewer setup')
            print('Viewer flags before: ', self.viewer.vopt.flags)
        """ viewer flags:
        0 - convex hull
        1 - textures
        2 - joints
        3 - actuators
        4 - camera
        9 - inertia boxes
        11 - perturbation force
        12 - perturbation obj
        13 - contact points
        14 - contact forces
        15 - split contact force into normal and tangent, 14 needs to be enabled first
        16 - make dynamic geoms more transparent
        17 - auto connect joints and body coms
        18 - body com
        20 - ground plane
        """
        self.viewer.vopt.flags[0] = 1
        self.viewer.vopt.flags[14] = self.viewer.vopt.flags[15] = 0
        self.viewer.vopt.flags[11] = self.viewer.vopt.flags[12] = 1
        if verbose:
            print('Viewer flags after:  ', self.viewer.vopt.flags)
        self.viewer.vopt.frame = 1  # body frames


    def evaluate_success(self, paths):
        num_success = 0
        num_stability = 0
        num_paths = len(paths)
        succ_string = '1'*50  # 50 consecutive timesteps of a successful grasp
        stability_string = '1'*300  # all timesteps during perturbation should satisfy the grasp success metric
        for path in paths:
            ep_succ_string = ''.join([str(int(ele)) for ele in path['env_infos']['obj_grab']])
            ep_stability_string = ''.join([str(int(ele)) for ele in path['perturb_infos']['obj_grab']])
            # succ if last 50 timesteps
            if ep_succ_string[-50:] == succ_string:
                num_success += 1
            if stability_string in ep_stability_string:
                num_stability += 1
        success_percentage = num_success * 100.0 / num_paths
        stability_percentage = num_stability * 100.0 / num_paths
        return success_percentage, stability_percentage

    
    def debug(self):
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        obj_com = self.data.subtree_com[self.obj_bid].ravel()
        body_xpos = self.data.body_xpos[self.obj_bid]
        body_xquat = self.data.body_xquat[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()

        print('obj_pos: ', obj_pos)
        print('obj_com: ', obj_com)
        print('body_xpos: ', body_xpos)
        print('body_xquat: ', body_xquat)
        print('geom_pos', self.model.geom_pos)
        print('geom_quat', self.model.geom_quat)
        print('palm_pos: ', palm_pos)
        print('Number of vertices: ', self.sim.model.mesh_vert.shape)
        print('Number of meshes: ', self.model.nmesh)
        print('Number of vertices in all meshes: ', self.model.nmeshvert)
        print('Number of vertices in each mesh: ', self.model.mesh_vertnum)
        print('Starting address of each mesh: ', self.model.mesh_vertadr)

        self.get_multicontact_verts(verbose=True)

        self.mj_viewer_setup()
        cnt = 0
        while True:
            if cnt%200 == 0:
                self.reset_model()
                palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
                contact_verts = self.get_multicontact_verts()
                hand_multi_pos = self.get_multihand_pos(self.touch_sensor_names)
                print(len(contact_verts), len(hand_multi_pos))
                hand_array = np.repeat(hand_multi_pos, self.n_verts_multicontact, axis=0)
                multicontact_array = np.tile(contact_verts, (len(hand_multi_pos), 1))
                print((hand_array-multicontact_array).ravel().shape)
                chamfer_dist = self.chamfer_dist(contact_verts, hand_multi_pos)
                print('multicontact chamfer dist: ', chamfer_dist)
                print()
            self.viewer.add_marker(pos=obj_pos, rgba=[0.2, 0.2, 0.8, 1.0],
                                   size=[0.015, 0.015, 0.015], label="")
            contact_verts = self.get_multicontact_verts()
            for vert in contact_verts:
                self.viewer.add_marker(pos=vert, rgba=[0.0, 0.8, 0.2, 1.0],
                                  size=[0.01, 0.01, 0.01], label="")
            hand_multi_pos = self.get_multihand_pos(self.touch_sensor_names)
            for touch_pos in hand_multi_pos:
                self.viewer.add_marker(pos=touch_pos, rgba=[0.8, 0.2, 0.8, 1.0],
                                  size=[0.01, 0.01, 0.01], label="")
            self.viewer.render()
            cnt += 1


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj', type=str, help='Type of object. '
                                                'Choose from [hammer, knife, mug, pan, toothbrush, teapot]')
    parser.add_argument('--multicontact', type=str, default='percentage', help='Type of object. Choose from [threshold, percentage]')
    parser.add_argument('--multivalue', type=float, default=0.5, help='Value for the multi-contact parameter.')
    parser.add_argument('--obj_rot', action='store_true', help='No object rotation')
    parser.add_argument('--gpu-env', type=str, default='0', help='gpu id to render env on')
    args = parser.parse_args()
    grasp_attrs_dict = {'dataset': 'contactdb',
                        'obj': args.obj,
                        'policy': 'mlp',
                        'noise': False,
                        'cameras': 'free',
                        'rewards': 'grasp:1',
                        'reward_dst_thr': 0.5,
                        'obj_mass': 0.2,
                        'obj_rot': args.obj_rot,
                        'obj_tr': False,
                        'gravity': 9.8,
                        'debug': False}
    obj = GraffV0(grasp_attrs_dict=grasp_attrs_dict, object='pan', device_id=int(args.gpu_env))
    obj.debug()
