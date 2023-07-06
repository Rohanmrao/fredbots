import rospy
import math
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
import torch as T
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class SharedAdam(T.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
            weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps,
                weight_decay=weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = T.zeros_like(p.data)
                state['exp_avg_sq'] = T.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, gamma=0.99):
        super(ActorCritic, self).__init__()

        self.gamma = gamma

        self.pi1 = nn.Linear(*input_dims, 128)
        self.v1 = nn.Linear(*input_dims, 128)
        self.pi = nn.Linear(128, n_actions)
        self.v = nn.Linear(128, 1)

        self.rewards = []
        self.actions = []
        self.states = []

    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def forward(self, state):
        pi1 = F.relu(self.pi1(state))
        v1 = F.relu(self.v1(state))

        pi = self.pi(pi1)
        v = self.v(v1)

        return pi, v

    def calc_R(self, done):
        states = T.tensor(self.states, dtype=T.float)
        _, v = self.forward(states)

        R = v[-1]*(1-int(done))

        batch_return = []
        for reward in self.rewards[::-1]:
            R = reward + self.gamma*R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = T.tensor(batch_return, dtype=T.float)

        return batch_return

    def calc_loss(self, done):
        states = T.tensor(self.states, dtype=T.float)
        actions = T.tensor(self.actions, dtype=T.float)

        returns = self.calc_R(done)

        pi, values = self.forward(states)
        values = values.squeeze()
        critic_loss = (returns-values)**2

        probs = T.softmax(pi, dim=1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -log_probs*(returns-values)

        total_loss = (critic_loss + actor_loss).mean()
    
        return total_loss

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float)
        pi, v = self.forward(state)
        probs = T.softmax(pi, dim=1)
        dist = Categorical(probs)
        action = dist.sample()


class Agent(mp.Process):
    def __init__(self, global_actor_critic, optimizer, input_dims, n_actions, 
                gamma, lr, name, global_ep_idx, env_id):
        super(Agent, self).__init__()
        self.local_actor_critic = ActorCritic(input_dims, n_actions, gamma)
        self.global_actor_critic = global_actor_critic
        self.name = 'w%02i' % name
        self.episode_idx = global_ep_idx
        self.prev_x = 0  # Previous x-coordinate of the turtle
        self.prev_y = 0  # Previous y-coordinate of the turtle
        self.velocity_publisher = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)
        self.pose_subscriber = rospy.Subscriber('/turtle1/pose', Pose, self.update_pose)
        self.optimizer = optimizer

    def update_pose(self, data):
        self.prev_x = data.x
        self.prev_y = data.y

    def run(self):
        t_step = 1
        while self.episode_idx.value < N_GAMES:
            done = False
            self.prev_x = 0  # Reset the previous x-coordinate for each episode
            self.prev_y = 0  # Reset the previous y-coordinate for each episode
            score = 0
            self.local_actor_critic.clear_memory()

            # Manually set the turtle's initial position
            init_pose = Twist()
            init_pose.linear.x = 5.0
            init_pose.angular.z = 0.0
            self.velocity_publisher.publish(init_pose)

            while not done:
                # Calculate the Euclidean distance between current position and previous position
                distance = math.sqrt((self.prev_x - self.pose.x)**2 + (self.prev_y - self.pose.y)**2)
                score += distance  # Use distance as the reward
                self.prev_x = self.pose.x
                self.prev_y = self.pose.y

                # Publish the action as a Twist message
                action = self.local_actor_critic.choose_action([self.pose.x, self.pose.y])
                velocity_message = Twist()
                velocity_message.linear.x = action  # Use action as the linear velocity
                velocity_message.angular.z = 0.0
                self.velocity_publisher.publish(velocity_message)

                # Check termination conditions
                # ...

            with self.episode_idx.get_lock():
                self.episode_idx.value += 1
            print(self.name, 'episode ', self.episode_idx.value, 'reward %.1f' % score)

if __name__ == '__main__':
    rospy.init_node('turtle_rl_node', anonymous=True)

    lr = 1e-4
    n_actions = 2
    input_dims = [2]  # Assuming the observation space consists of x and y coordinates
    N_GAMES = 3000
    T_MAX = 5
    global_actor_critic = ActorCritic(input_dims, n_actions)
    global_actor_critic.share_memory()
    optim = SharedAdam(global_actor_critic.parameters(), lr=lr, 
                        betas=(0.92, 0.999))
    global_ep = mp.Value('i', 0)

    workers = [Agent(global_actor_critic,
                    optim,
                    input_dims,
                    n_actions,
                    gamma=0.99,
                    lr=lr,
                    name=i,
                    global_ep_idx=global_ep,
                    env_id=env_id) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    [w.join() for w in workers]
