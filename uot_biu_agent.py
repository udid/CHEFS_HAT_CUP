import gym
from gym import logger as gymlogger
from gym.wrappers import Monitor
gymlogger.set_level(40) # error only

import numpy as np
import random

from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os

from ChefsHatGym.Agents import IAgent, Agent_Naive_Random
from ChefsHatGym.Rewards import RewardOnlyWinning
from ChefsHatGym.env import ChefsHatEnv
from ChefsHatGym.Evaluation import Tournament


##################################################################################################
class MyReward():
    rewardName = "MyReward"
    def getReward(self, thisPlayerPosition, performanceScore, matchFinished):
        reward = - 0.001
        if matchFinished:
            #finalPoints = (3 - thisPlayerPosition)/3
            #reward = finalPoints + performanceScore
            if thisPlayerPosition == 0:
                reward = 1
        return reward

class ExperienceReplay:
    def __init__(self, experience_buffer_max_size=100000, experience_buffer_min_size=1000):
        self._max_size    = experience_buffer_max_size
        self._min_size    = experience_buffer_min_size
        self._memory      = deque(maxlen = self._max_size)

    def append(self, inst):
        self._memory.append(inst)

    def sample(self, batch_size):
        if len(self._memory) < self._min_size: 
            return None
        return random.sample(self._memory ,batch_size)


class MyModule(nn.Module):
    def __init__(self, cfg):
        super(MyModule, self).__init__()
        self._fc1= nn.Linear(cfg['status_size'],    cfg['model_fc1_size'])
        self._fc2= nn.Linear(cfg['model_fc1_size'], cfg['model_fc2_size'])
        self._fc3= nn.Linear(cfg['model_fc2_size'], cfg['actions_size'])

    def forward(self, input):
        v = F.relu(self._fc1(input))
        v = F.relu(self._fc2(v))
        v = self._fc3(v)
        return v

class MyDqnAgent:
    def __init__(self, cfg, learning_agent ,shared_experience_replay=None):
        self._policy_network           = MyModule(cfg)
        self._target_network           = MyModule(cfg)
        if learning_agent:
            self._experience_replay        = ExperienceReplay() if shared_experience_replay is None  else shared_experience_replay
        else:
            self._experience_replay        = ExperienceReplay(100, 10)
        self._update_policy_interval   = cfg['update_policy_interval']
        self._update_target_interval   = cfg['update_target_interval']
        
        self._actions_size             = cfg['actions_size']
        self._gamma                    = cfg['gamma']
        self._tau                      = cfg['tau']
        self._batch_size               = cfg['batch_size']
        self._optimizer                = optim.Adam(self._policy_network.parameters(), lr=cfg['optimizer_learning_rate'])

    def select_action(self, state, possible_actions, exploration_exploitation):
        action_id = 0

        possible_actions_list = np.array(np.where(np.array(possible_actions) == 1))[0].tolist()
        
        if len(possible_actions_list) == 1: 
            #print('action: no chance')
            action_id = possible_actions_list[0]
        elif random.random() < exploration_exploitation:
            #print('action: random')
            if 199 in possible_actions_list: possible_actions_list.remove(199)
            random.shuffle(possible_actions_list)
            action_id = possible_actions_list[0]
        else:
            if 199 in possible_actions_list: possible_actions_list.remove(199)
            #print('action: main')
            state_vec = torch.from_numpy(state).float().unsqueeze(0)
            self._policy_network.eval()
            with torch.no_grad():
                state_q_values = self._policy_network(state_vec)
            self._policy_network.train()
          
            v = state_q_values.numpy()[0][possible_actions_list]
            i = np.arange(self._actions_size)[possible_actions_list]
            action_id = i[np.argmax(v)]
          
        return action_id
    
    def step(self, time_id, state, action_id, reward, next_state, is_done):
        self._experience_replay.append((state, action_id, reward, next_state, 1 if is_done else 0))

        if time_id % self._update_policy_interval == 0 :
            #print('update_policy', time_id)
            self.update_policy()

        if time_id % self._update_target_interval == 0:
            #print('update_target', time_id)
            self.update_target()

    def update_target(self):
        for t, p in zip(self._target_network.parameters(), self._policy_network.parameters()):
            t.data.copy_(self._tau*p.data + (1.0-self._tau)*t.data)

    def update_policy(self):
        batch = self._experience_replay.sample(self._batch_size)
        if batch is None: return   
    
        replay_header = {'states':torch.float32,
                         'action_ids':torch.int64,
                         'rewards':torch.float32,
                         'next_states':torch.float32,
                         'is_dones':torch.float32}
        batch_tensors = { k : list() for k in replay_header}
        
        for batch_items in batch:
            for i, name  in enumerate(replay_header):
                batch_tensors[name].append(batch_items[i])
            
        for name, to_type in replay_header.items():
            batch_tensors[name] = torch.from_numpy(np.vstack(batch_tensors[name])).to(to_type)
    
        next_state_q = self._target_network(batch_tensors['next_states']).detach().max(1)[0].unsqueeze(1)
        
        q     = batch_tensors['rewards'] + (self._gamma * next_state_q * (1 - batch_tensors['is_dones']))
        q_hat = self._policy_network(batch_tensors['states']).gather(1, batch_tensors['action_ids'])

        loss = F.mse_loss(q_hat, q)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
      
    def save_policy(self, file_path):
        print('saving policy to: ', file_path )
        torch.save(self._policy_network.state_dict(), file_path)

    def load_policy(self, file_path):
        print('loading policy from: ', file_path)
        self._policy_network.load_state_dict(torch.load(file_path))

class UOT_BIU_Agent(IAgent.IAgent):
    def __init__(self, name="NAIVE", policy_file_path = 'policy.pth', learning_agent=False, shared_experience_replay=None):
        #####################################################
        ##################### CONFIG ########################
        #####################################################
        cfg = { 
            
            # env
            'status_size' : 28,  
            'actions_size': 200,
            
            # exploration exploitation
            'epsilon_start_value': 1.0,
            'epsilon_min_value'  : 0.2,
            'epsilon_decay'      : 0.995,

            # experience replay cfg
            'experience_buffer_max_size': 100000,
            'experience_buffer_min_size': 1000,
            
            # network params
            'model_fc1_size': 64, 
            'model_fc2_size': 64, 
            
            # learning params
            'update_policy_interval' : 4,
            'update_target_interval' : 100,
            'gamma'                  : 0.995,
            'tau'                    : 0.001,
            'batch_size'             : 64,
            'optimizer_learning_rate': 0.0005,

            
            }

        
        self.name = "UOT_BIU_"+str(name)
        self.learning_agent = learning_agent
        self.module = MyDqnAgent(cfg, learning_agent, shared_experience_replay)
        self.reward = MyReward()
        self.time_id = 0
        self.match_counter = 0
        
        self.epsilon           = cfg['epsilon_start_value']
        self.epsilon_decay     = cfg['epsilon_decay']
        self.epsilon_min_value = cfg['epsilon_min_value']

        self.policy_file_path = policy_file_path
        
        if not self.learning_agent and (policy_file_path is None or not os.path.exists(self.policy_file_path)):
            msg = "could not find the file:" + str(policy_file_path) + " please use the policy_file_path arg in the agent contractor. " 
            print(msg)
            raise Exception(msg)
            
        self.load_policy()

    def save_policy(self, file_path):
        self.module.save_policy(file_path)
  
    def load_policy(self):
        if self.policy_file_path != '' and os.path.exists(self.policy_file_path ): 
            self.module.load_policy(self.policy_file_path)

    #####################################################
    ##################### IAgent ########################
    #####################################################
    def getAction(self,  observations):
        curr_state, curr_possible_actions = observations[:28],  observations[28:]
        epsilon = self.epsilon if self.learning_agent else 0
        
        action_id = self.module.select_action(curr_state, curr_possible_actions, epsilon)
        
        ret = np.zeros(200)
        ret[action_id] = 1

        return ret

    def actionUpdate(self, observations, nextobs, action, reward, info):
        if not self.learning_agent: return 
        #reward = getReward(info,None,None)
        curr_state, curr_possible_actions = observations[:28], observations[28:]
        next_state, next_possible_actions = nextobs[:28], nextobs[28:]
        action_id = np.argmax(action)
        
        is_done = info["thisPlayerFinished"]

        self.time_id += 1
        
        self.epsilon *=  self.epsilon_decay
        if self.epsilon < self.epsilon_min_value: self.epsilon = self.epsilon_min_value

        self.module.step(self.time_id ,curr_state, action_id, reward, next_state, is_done);

    def observeOthers(self, envInfo):
        pass

    def getReward(self, info, stateBefore, stateAfter):
        thisPlayer = info["thisPlayerPosition"]
        matchFinished = info["thisPlayerFinished"]
        thisPlayerPerformanceScore = info["performanceScore"][info['thisPlayer']]
        return self.reward.getReward(thisPlayer, thisPlayerPerformanceScore, matchFinished)






    def run_agent_trainer():
            
        """Game parameters"""
        gameType = ChefsHatEnv.GAMETYPE["MATCHES"]
        gameStopCriteria = 20
        rewardFunction = RewardOnlyWinning.RewardOnlyWinning()

        shared_experience_replay  = ExperienceReplay()
        """Player Parameters"""
        agent1 = UOT_BIU_Agent("learning_agent1", policy_file_path = 'policy.pth', learning_agent=True, shared_experience_replay=shared_experience_replay)
        agent2 = UOT_BIU_Agent("learning_agent2", policy_file_path = 'policy.pth', learning_agent=True, shared_experience_replay=shared_experience_replay)
        agent3 = UOT_BIU_Agent("learning_agent3", policy_file_path = 'policy.pth', learning_agent=True, shared_experience_replay=shared_experience_replay)
        agent4 = UOT_BIU_Agent("learning_agent4", policy_file_path = 'policy.pth', learning_agent=True, shared_experience_replay=shared_experience_replay)
        
        playersAgents = [agent1, agent2, agent3, agent4]
        agentNames    = [agent.name      for agent in playersAgents]
        rewards       = [agent.getReward for agent in playersAgents]
        
        """Experiment parameters"""
        saveDirectory = "log_folder"
        if not os.path.exists(saveDirectory): os.makedirs(saveDirectory)
        verbose = False
        saveLog = False
        saveDataset = False
        game_to_play = 1000
        seed = 1234
        
        """Setup environment"""
        env = gym.make('chefshat-v0') #starting the game Environment
        env.startExperiment(rewardFunctions=rewards, gameType=gameType, stopCriteria=gameStopCriteria, playerNames=agentNames, logDirectory=saveDirectory, verbose=verbose, saveDataset=True, saveLog=True)

        env.seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        """Start Environment"""
        p = None
        for _ in range(game_to_play):
            observations = env.reset()
            while not env.gameFinished:
                currentPlayer = playersAgents[env.currentPlayer]
                observations = env.getObservation() #228 size
            
                info = {"validAction":False}
                while not info["validAction"]:
                    action = currentPlayer.getAction(observations)
                    nextobs, reward, isMatchOver, info = env.step(action)
                currentPlayer.actionUpdate( observations, nextobs, action, reward, info)

                if isMatchOver:
                    currentPlayer.matchUpdate(info)
                    print ("-------------")
                    print ("Match:" + str(info["matches"]))
                    print ("Score:" + str(info["score"]))
                    print ("Performance:" + str(info["performanceScore"]))
                    print ("-------------")
                    p = info["performanceScore"]
          
                if p is not None: 
                    pa = playersAgents[p.index(max(p))]
                    pa.save_policy('policy.pth')
                    # the worse player policy will be replaced by the best  
                    playersAgents[p.index(min(p))].load_policy()

def run_tournament_for_example():
    playersAgents = [UOT_BIU_Agent("1")]
    for i in range(2,5):
        playersAgents.append(Agent_Naive_Random.AgentNaive_Random(str(i)))

    tournament = Tournament.Tournament('tournament', opponentsComp=playersAgents,  oponentsCompCoop=[], verbose=True, threadTimeOut=5, actionTimeOut=5, gameType=ChefsHatEnv.GAMETYPE["MATCHES"], gameStopCriteria=30)
    tournament.runTournament()