import torch
import numpy as np
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.atari_wrappers import AtariWrapper
from betty.utils import neg_with_none, to_vec
from resources.SoBiRL.sac_subroutine import SAC
from resources.SoBiRL.reward_util.reward_construction import CusCnnRewardNet
from resources.SoBiRL.wrapper_util.env_sample_wrapper import AutoResetWrapper

def construct_env(env_id, seed=1):
    """Construct the RL environment with appropriate wrappers."""
    atari_env = gym.make(env_id, render_mode="rgb_array")
    atari_env = gym.wrappers.RecordEpisodeStatistics(atari_env)
    preprocessed_env = AtariWrapper(atari_env)
    endless_env = AutoResetWrapper(preprocessed_env)
    return endless_env

def reinforce(vector, curr, prev, sync):
    """
    Approximate the matrix-vector multiplication with the best response Jacobian using the
    REINFORCE method with RL environment and agent. This method is particularly useful for
    bilevel optimization problems where the lower-level problem involves RL.

    :param vector:
        Vector with which matrix-vector multiplication with best-response Jacobian (matrix) would
        be performed.
    :type vector: Sequence of Tensor
    :param curr: A current level problem
    :type curr: Problem
    :param prev: A directly lower-level problem to the current problem
    :type prev: Problem
    :param sync: Whether to synchronize gradients
    :type sync: bool
    :return: (Intermediate) gradient
    :rtype: Sequence of Tensor
    """
    config = curr.config
    
    # Initialize RL environment and agent if not already done
    if not hasattr(curr, 'env'):
        # Create vectorized environment
        env = make_vec_env(lambda: construct_env(config.env_id), seed=config.seed)
        env = VecFrameStack(env, n_stack=4)
        curr.env = env
        
        # Initialize reward network
        reward_net = CusCnnRewardNet(
            env.observation_space,
            env.action_space,
        ).to(config.device)
        
        # Initialize SAC agent
        curr.agent = SAC(
            reward_model=reward_net,
            env=env,
            args=config,
        )
    
    # Get the current loss and parameters
    in_loss = curr.training_step_exec(curr.cur_batch)
    curr_params = curr.trainable_parameters()
    prev_params = prev.trainable_parameters()
    
    # Sample trajectories using the RL agent
    with torch.no_grad():
        num_samples = config.reinforce_samples if hasattr(config, 'reinforce_samples') else 10
        sampled_gradients = []
        
        for _ in range(num_samples):
            # Generate a trajectory using the current policy
            obs = curr.env.reset()
            done = False
            total_reward = 0
            
            while not done:
                # Get action from current policy
                action, _, _ = curr.agent.actor.get_action(torch.Tensor(obs).to(config.device))
                action = action.detach().cpu().numpy()
                
                # Step environment
                next_obs, reward, done, info = curr.env.step(action)
                total_reward += reward
                
                # Update observation
                obs = next_obs
            
            # Compute gradient of the trajectory's total reward
            sampled_loss = torch.tensor(total_reward, requires_grad=True)
            sampled_grad = torch.autograd.grad(
                sampled_loss, curr_params, create_graph=True
            )
            
            # Compute the REINFORCE gradient estimate
            reinforce_grad = []
            for g, v in zip(sampled_grad, vector):
                if g is not None and v is not None:
                    reinforce_grad.append(g * v)
                else:
                    reinforce_grad.append(None)
            
            sampled_gradients.append(reinforce_grad)
    
    # Average the sampled gradients
    avg_grad = []
    for i in range(len(vector)):
        valid_grads = [g[i] for g in sampled_gradients if g[i] is not None]
        if valid_grads:
            avg_grad.append(torch.stack(valid_grads).mean(0))
        else:
            avg_grad.append(None)
    
    # Scale by the learning rate
    if hasattr(config, 'reinforce_lr'):
        avg_grad = [g * config.reinforce_lr if g is not None else None for g in avg_grad]
    
    if sync:
        avg_grad = [neg_with_none(g) for g in avg_grad]
        torch.autograd.backward(
            in_loss, inputs=prev_params, grad_tensors=avg_grad
        )
        implicit_grad = None
    else:
        implicit_grad = torch.autograd.grad(
            in_loss, prev_params, grad_outputs=avg_grad
        )
        implicit_grad = [neg_with_none(ig) for ig in implicit_grad]
    
    return implicit_grad
