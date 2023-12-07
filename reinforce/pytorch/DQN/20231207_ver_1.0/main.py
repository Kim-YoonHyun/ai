import sys
import os
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from Gridworld import Gridworld

from models import networks as net

# share modules
sys.path.append('/home/kimyh/python/ai')
from sharemodule import trainutils as tum

# get action
def get_action(action_set, qval, epsilon):
    qval = qval.cpu().data.numpy()
    if random.random() < epsilon:
        action_idx = np.random.randint(0, 4)
    else:
        action_idx = np.argmax(qval)
    action = action_set[action_idx]
    return action_idx, action


def main():
    epochs = 1000
    learning_rate = 1e-3    
    gamma = 0.9
    epsilon = 1.0
    
    action_set = {
        0:'u',
        1:'d',
        2:'l',
        3:'r'
    }
    
    # device
    device = tum.get_device(0)
    
    # model
    model = net.DQN(
        input_num=64,
        output_num=4
    )
    model.to(device)
    
    # optimizer
    optimizer = tum.get_optimizer(
        base='torch',
        method='Adam',
        model=model,
        learning_rate=learning_rate
    )
    # loss function
    loss_function = tum.get_loss_function(
        base='torch',
        method='MSE'
    )
    
    # train    
    losses = []
    for _ in tqdm(range(epochs)):
        game = Gridworld(size=4, mode='static')
        position_state = game.board.render_np().reshape(1, 64)
        random_state = np.random.rand(1, 64)/10.0
        now_state = position_state + random_state
        now_state = torch.from_numpy(now_state).float().to(device)
        
        while True:
            # t
            q_t = model(now_state)
            action_idx, action = get_action(
                action_set=action_set, 
                qval=q_t,
                epsilon=epsilon
            )
            x = q_t.squeeze()[action_idx]
            
            # player 를 action 방향으로 이동
            game.make_move(action) 
            
            # t + 1
            position_state = game.board.render_np().reshape(1, 64)
            random_state = np.random.rand(1, 64)/10.0
            next_state = position_state + random_state
            next_state = torch.from_numpy(next_state).float().to(device)
            
            # reward 
            reward = game.reward()
            with torch.no_grad():
                q_t_1 = model(next_state)
            q_t_1_max = torch.max(q_t_1)
            
            # 벨만 방정식
            if reward == -1:
                y = reward + (gamma*q_t_1_max) 
            else:
                y = reward
            y = torch.tensor([y]).detach().float().to(device)
            loss = loss_function(x, y)
            
            # gradient
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            
            # 진행
            now_state = next_state
            if reward != -1:
                break
        if epsilon > 0.1:
            epsilon -= (1/epochs)
    return model, device


def test_model(model, device, mode='static', display=True):
    i = 0
    action_set = {
        0:'u',
        1:'d',
        2:'l',
        3:'r'
    }
    test_game = Gridworld(mode=mode)
    state_ = test_game.board.render_np().reshape(1, 64) + np.random.rand(1, 64)/10.0
    state = torch.from_numpy(state_).float().to(device)
    
    if display:
        print('Initial State:')
        print(test_game.display())
    status = 1
    while status == 1:
        qval = model(state)
        qval_ = qval.cpu().data.numpy()
        action_ = np.argmax(qval_)
        action = action_set[action_]
        if display:
            print(f'Move: {i} Taking action: {action}')
        test_game.make_move(action)
        state_ = test_game.board.render_np().reshape(1, 64) + np.random.rand(1, 64)/10.0
        state = torch.from_numpy(state_).float().to(device)
        if display:
            print(test_game.display())
        reward = test_game.reward()
        if reward != -1:
            if reward > 0:
                status = 2
                if display:
                    print(f'Game won! Reward:: {reward}')
            else:
                status = 0
                if display:
                    print(f'Game Lost. Reward: {reward}')
        i += 1
        if (i > 15):
            if display:
                print('Game Lost, too may moves')
            break
    
    if status == 2:
        win = True
    else:
        win = False
    return win
        

if __name__ == '__main__':
    model, device = main()
    test_model(model, device)