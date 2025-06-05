import gym
import numpy as np
import pybullet as p
import random
from supertrack.resources.character import Character
from supertrack.resources.character_k import Character_k
from supertrack.resources.plane import Plane
import time
from gym import spaces
from pybullet_utils import bullet_client as bc
from bvh import Bvh

# random.seed(10) #! stochasticity
class SuperTrackEnv(gym.Env):
    metadata = {'render_modes': ['human']}  
  
    def __init__(self, render_mode=None):
        print('sup')

        self.render_mode = render_mode 
        # spaces
        state_s = spaces.Dict({
        'pos': spaces.Box(low=-10, high=10, shape=(22, 3), dtype=np.float32),
        'vel': spaces.Box(low=-10, high=10, shape=(22, 3), dtype=np.float32),
        'rot': spaces.Box(low=-10, high=10, shape=(22, 4), dtype=np.float32),
        'rot_vel': spaces.Box(low=-10, high=10, shape=(22, 3), dtype=np.float32)
        })

        state_k = spaces.Dict({
        'pos': spaces.Box(low=-10, high=10, shape=(22, 3), dtype=np.float32),
        'vel': spaces.Box(low=-10, high=10, shape=(22, 3), dtype=np.float32),
        'rot': spaces.Box(low=-10, high=10, shape=(22, 4), dtype=np.float32),
        'rot_vel': spaces.Box(low=-10, high=10, shape=(22, 3), dtype=np.float32)
        })
        
        self.observation_space = spaces.Dict({
            'state_s': state_s,
            'state_k': state_k
        })
        
        self.action_space = spaces.Dict({
            'rot': spaces.Box(low=-10, high=10, shape=(22, 4), dtype=np.float32),
            # 'rot_vel': spaces.Box(low=-10, high=10, shape=(22, 3), dtype=np.float32)
            # you cant get rot_vel from kinematic character which uses resetJointStateMultiDof to move
        })
        
        # self.action_space = spaces.Box(low=-10, high=10, shape=(63,), dtype=np.float32) # action space 
        # self.observation_space =  spaces.Box(low=-10, high=10, shape=(710,), dtype=np.float32) # observation space X = {x 𝑝os,  x vel,  x 𝑟ot,  x 𝑟otVel }

        # variables
        self.plane = None
        self.character = None
        self.character_k = None
        self.mocap_data = None
        self.done = False
        self.frame_i = 0
        self.num_frames = None
        self.dt = 1./60. # 0.016
        self.step_i = self.dt/(1./30.) # 
        # self.state_0 = None 
        # self.state_k_0 = None

        if render_mode == 'human':
            self.p0 = bc.BulletClient(connection_mode=p.GUI) # GUI setup
            self.p0.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
        
            self.p0.resetDebugVisualizerCamera(cameraDistance=5,
                            cameraYaw=0,
                            cameraPitch=-20,
                            cameraTargetPosition=[0, 0, 0])
        else:
            self.p0 = bc.BulletClient(connection_mode=p.DIRECT)
            self.p0.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
        
        # self.p0.setRealTimeSimulation(1)
        self.p0.setPhysicsEngineParameter(numSolverIterations=100, fixedTimeStep=self.dt, numSubSteps=4) # number of solver iterations/ stuff like colision detection are monitored more often with more iteration / simulation update 
        # self.p0.setPhysicsEngineParameter(fixedTimeStep=self.dt,
        #                     solverResidualThreshold= -9, # velocity threshold
        #                     numSolverIterations=100, # detection accuracy
        #                     numSubSteps=1) # further TimeStep subdivision

        
        if self.p0.getPhysicsEngineParameters() is None:
            print("Errorrrrrrrr: Not connected to the physics server.")
        else:
            print("Physicssssss server connected.")



    def to_local(self, state): #
        """
        all_pos  
        all_vel
        all_rot
        all_rot_vel
        """

        local_pos = []  
        local_vel = []
        local_rot = []
        local_rot_vel = []
        local_height = []

        #----------------------------- pre-processing steps -----------------------------
        """
        The global space coordinates are inappropriate to provide to a network directly 
        as they are not translation or rotation invariant, and the raw quaternions have issues of
        double-cover. 

        """

        #----------------------------- Case with heading -----------------------------
        root_wp = state['pos'][0] # hip pos
        root_wr = state['rot'][0]
        # inv_root_wp = [-root_wp[0], -root_wp[1], -root_wp[2]]
        # heading_r = self.buildHeadingTrans(root_wr) 
        
        # # root of the character is transformed so that its forward direction matches the world’s X-axis.
        # inv_orig_p, inv_orig_r = self.p0.multiplyTransforms([0, 0, 0],
        #                                                         heading_r,
        #                                                         inv_root_wp,
        #                                                         [0, 0, 0, 1])

        # same as inv_orig_p = self.rotate_vector_by_quaternion(heading_r, inv_root_wp)

        inv_root_wp, inv_root_wr  = p.invertTransform(root_wp, root_wr) 
        # print('inv', inv_root_wp)
        # root_lp, root_lr = self.p0.multiplyTransforms(inv_root_wp,  
        #                                         inv_root_wr,
        #                                         root_wp, 
        #                                         root_wr)
        for i in range(len(state['pos'])):
            
            # link_lp, link_lr = self.p0.multiplyTransforms(
            #                             inv_root_wp,    
            #                             inv_root_wr, 
            #                             state['pos'][i], 
            #                             state['rot'][i])
            # link_lp_t, link_lr_t = self.p0.multiplyTransforms(
            #                             inv_root_wp,    
            #                             inv_root_wr, 
            #                             state['pos'][i], 
            #                             [0,0,0,1])
            # link_lp = [ # pos relative to the root
            #     link_lp[0] - root_lp[0], link_lp[1] - root_lp[1],
            #     link_lp[2] - root_lp[2]
            # ]
            # print(state['pos'][i])
            # print(state['rot'][i])
            # print("link local pos", link_lr)
            # print("test", link_lr_t)
            # print("link local rot", link_lr)
            # local_pos.append(link_lp)
            # # local_rot.append(link_lr)

            # link_lv, _ = self.p0.multiplyTransforms([0, 0, 0], 
            #                                         inv_orig_r,
            #                                         state['vel'][i],
            #                                         [0, 0, 0, 1])

            # link_lav, _ = self.p0.multiplyTransforms([0, 0, 0], 
            #                                          inv_orig_r,
            #                                          state['rot_vel'][i], 
            #                                          [0, 0, 0, 1])
            # local_vel.append(link_lv)
            # local_rot_vel.append(link_lav)



            # print('\n1111111111111111\nlinkPosLocal',linkPosLocal)
            # print('\nlinkOrnLocal', )
            # print('\nlinkLinVelLocal',linkLinVelLocal)            
            # print('\nlinkAngVelLocal',linkAngVelLocal)
            # print(state['pos'][i])
            #-------------------------------------------------------------------------
            transformed_pos = self.p0.rotateVector(inv_root_wr, (np.array(state['pos'][i]) - np.array(state['pos'][0])))
            local_pos.append(transformed_pos)

            transformed_vel = self.p0.rotateVector(inv_root_wr, state['vel'][i])
            local_vel.append(transformed_vel)

            _, transformed_rot = self.p0.multiplyTransforms([0,0,0], inv_root_wr, [0,0,0], state['rot'][i]) # quaternion mul
            transformed_rot = self.p0.getMatrixFromQuaternion(transformed_rot)
            transformed_rot = np.array(transformed_rot).reshape(3, 3)
            transformed_rot = transformed_rot[:, [0, 2]].flatten()  # Extract X (forward) & Z (sideways) from a 3×3 rotation matrix
            local_rot.append(transformed_rot) # two-axis rotation matrix as 6x vector
            # print('\nROT_ROT_ROT NP',transformed_rot)
            transformed_rot_vel = self.p0.rotateVector(inv_root_wr, state['rot_vel'][i])
            local_rot_vel.append(transformed_rot_vel)

            # print('POS_POS_POS')
            # print(transformed_pos)
            # print('VEL_VEL_VEL')
            # print(transformed_vel)
            # print('ROT_ROT_ROT')
            # print(transformed_rot)
            # print('ANG_VEL_ANG_VEL')
            # print(transformed_rot_vel)

            # time.sleep(10000)
            # plane_pos, _ = self.p0.getBasePositionAndOrientation(self.plane.plane)
            # link_height = abs(state['pos'][i][1] - plane_pos[1])
            link_height = state['pos'][i][1]
            local_height.append(link_height)

            up_vec = self.p0.rotateVector(inv_root_wr, [0, 1, 0])
        return [local_pos, local_vel, local_rot, local_rot_vel, local_height, up_vec]

    def to_dict(self, state, state2):
        state_s = {
            'pos': np.array(state[0], dtype=np.float32),
            'vel': np.array(state[1], dtype=np.float32),
            'rot': np.array(state[2], dtype=np.float32),
            'rot_vel': np.array(state[3], dtype=np.float32),
        }
        state_k = {
            'pos': np.array(state2[0], dtype=np.float32),
            'vel': np.array(state2[1], dtype=np.float32),
            'rot': np.array(state2[2], dtype=np.float32),
            'rot_vel': np.array(state2[3], dtype=np.float32),
        }
        states = {
            'state_s': state_s,
            'state_k': state_k,
        }
        return states

    def reset(self):
        self.p0.resetSimulation() # removes all objects ?
        self.p0.setGravity(0, -9.8, 0)
        
        # sample our animation database for a random pose
        motion_file = ['walk3_subject4.bvh'] #  'walk1_subject2.bvh', 'walk1_subject5.bvh'
        motionPath = "SuperTrack/supertrack/mocap" + "/" + random.choice(motion_file)
        with open(motionPath, 'r') as f:
            mocap = Bvh(f.read())
        self.mocap_data = [[float(value) for value in frame] for frame in mocap.frames]
        # print(self.mocap_data[:2])
        # time.sleep(1000)
        self.num_frames = mocap.nframes
        rand_frame = random.randint(0, self.num_frames-int(np.ceil(50 * self.step_i))) # +2 cuz kin +1 and interpolation
        # rand_frame = random.randint(0, 100)

        # init models in rand_frame pose
        self.plane = Plane(self.p0)
        self.character_k = Character_k(self.p0, self.mocap_data, rand_frame+self.step_i) # initial target state
        self.character = Character(self.p0, self.mocap_data, rand_frame) # initial sim state

        # Disable collisions between every pair of links in character_1 and character_2
        for link_index_1 in range(-1, (self.p0.getNumJoints(self.character.humanoid))):  # -1 for the base link
            for link_index_2 in range(-1, (self.p0.getNumJoints(self.character_k.humanoid))):
                self.p0.setCollisionFilterPair(self.character.humanoid, self.character_k.humanoid, link_index_1, link_index_2, enableCollision=0)

        # get states
        state_s = self.character.get_state() 
        # state_k = self.character_k.get_state() #!!!!!!!!!!!!!!
        state_k = self.character_k.kin_state
        # print(state_k)
        states = self.to_dict(state_s, state_k)
        
        # print('####P####\n',state_s[0])
        # print('####R####\n',state_s[1])
        # print('####V####\n',state_s[2])
        # print('####RV####\n',state_s[3])
        self.currentTime = time.time() #!--
        # self.accumulator = 0

        self.frame_i = rand_frame
        self.done = False
        # info = {"k_targets": self.character_k.k_targets}
        info = {}
        return states, info

    def step(self, action): 
        action = action['rot']
        # print('Offfsets',action,'\n')
        action = np.array([self.p0.multiplyTransforms([0,0,0], q1, [0,0,0], q2)[1] for q1, q2 in zip(action, self.character_k.k_targets)]) #* final pd targets)
        # print(action)
        # print()
        # print('Targets',self.character_k.k_targets)
        # print('ACTION',action)
        # time.sleep(1000)
        # print("ACTION_ACTION_ACTION", action)
        # print("\nlen ",len(action))
        # time.sleep(1000)

        self.character.apply_action(action)

        self.frame_i += self.step_i
        self.character_k.set_frame(self.frame_i+self.step_i)
        self.p0.stepSimulation() # step simulation 
        # time.sleep(self.dt)
        #!-- maybe real time
        newTime = time.time()
        frameTime = newTime - self.currentTime
        # self.accumulator += frameTime
        # print("\nFRAME_TIME",frameTime)
        if (frameTime<self.dt):
            time.sleep((self.dt-frameTime))
            # time.sleep(5)
            # print("S")
        self.currentTime = newTime
        #!--
        # while (self.accumulator >= self.dt ):
        #     self.p0.stepSimulation()
        #     self.accumulator -= self.dt
            
        # return new states
        state_s = self.character.get_state() # 22 links/joints, action are 21
        state_k = self.character_k.get_state()
        states = self.to_dict(state_s, state_k)

        # print('####P####\n',state_s[0])
        # print('####R####\n',state_s[1])
        # print('####V####\n',state_s[2])
        # print('####RV####\n',state_s[3])
        # print("########state_s#########",state_s[0][0])
        # print("#",self.p0.getBasePositionAndOrientation(self.character))
        # ifs - reset
        head = np.array(state_s[0][5])
        head_k = np.array(state_k[0][5])
        dist = np.linalg.norm(head - head_k)
        if (self.frame_i+self.step_i*2 >= self.num_frames or dist > 0.95): # if dist is more then 25 cm     or dist > 0.25
            self.done = True

        reward = 0
        info = {'s_targets': action,
                'k_rot': self.character_k.k_targets}
        # print("@@@@@@@@@@@@@@@@@@@@@@@@@", self.character_k.k_targets)
        return states, reward, self.done, info
    
    def render(self):
        pass
    def close(self):
        pass
    def seed(self, seed=None): 
        pass    