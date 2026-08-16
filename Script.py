import pybullet as p
import pybullet_data
import numpy as np
import math
import random
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# КОНФИГУРАЦИЯ
MODE = "TRAIN"  # "TRAIN" или "TEST"
MODEL_FILE = "hexapod_ppo_best.pth"
CHECKPOINT_DIR = "checkpoints"
SAVE_EVERY_STEPS = 20000
TOTAL_STEPS_TO_TRAIN = 1000000
NORMALIZATION_WARMUP_STEPS = 20000
EVOLUTION_EVERY_STEPS = 20000

USE_MUTATION_BASED_SELECTION = True

NUM_LEADERS = 2
ELITE_COUNT = 1

CRITIC_MUTATION = 0.0
LOG_STD_MUTATION = 0.0

MAX_ACTOR_MUTATION = 0.01
MIN_ACTOR_MUTATION = 0.001

# Насколько увеличивать мутацию при застое
MUTATION_INCREASE_FACTOR = 1.5

# Насколько уменьшать мутацию после улучшения
MUTATION_DECREASE_FACTOR = 0.90

# После скольких поколений без улучшения усиливаем поиск
STAGNATION_LIMIT = 3


STATE_DIM = 41
ACTION_DIM = 18

ACTION_SCALE = 0.1

GAMMA = 0.99
GAE_LAMBDA = 0.95

LEARNING_RATE = 1e-5
PPO_EPOCHS = 10
MINIBATCH_SIZE = 64

VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
CLIP_EPSILON = 0.2

ROLLOUT_STEPS = 2048
MAX_EPISODE_STEPS = 2048
ACTION_REPEAT = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# СРЕДА (WORLD)
class World:
    def __init__(self):
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)
        p.setRealTimeSimulation(0)

        self.plane = p.loadURDF("plane.urdf")
        p.changeDynamics(self.plane, -1, lateralFriction=3.0, spinningFriction=1.5)

        self.target_speed = 0.0
        self.target_angle = 0.0
        self.action_repeat = ACTION_REPEAT

        # --- ИНИЦИАЛИЗАЦИЯ НОРМАЛИЗАЦИИ ---
        self.state_dim = STATE_DIM
        self.state_mean = np.zeros(STATE_DIM, dtype=np.float64)
        self.state_M2 = np.zeros(STATE_DIM, dtype=np.float64)
        self.state_std = np.ones(STATE_DIM, dtype=np.float64)

        self.count = 0
        self.eps = 1e-8
        self.normalize_frozen = False

    def freeze_normalization(self):
        self.normalize_frozen = True
        print("Нормализация заморожена")
        print("State mean:", self.state_mean)
        print("State std :", self.state_std)
    @staticmethod
    def normalize_angle(angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def update_state_stats(self, state_arr):

        if self.normalize_frozen:
            return

        state_arr = np.asarray(state_arr,dtype=np.float64)

        self.count += 1

        delta = state_arr - self.state_mean

        self.state_mean += (delta / self.count)

        delta2 = (state_arr - self.state_mean)

        self.state_M2 += (delta * delta2)

        if self.count > 1:
            variance = (self.state_M2 / (self.count - 1))
            self.state_std = np.sqrt(np.maximum(variance,1e-6))

    def get_state(self, robot, update_stats=True):
        state = []

        # СУСТАВЫ

        for i in range(18):
            joint = p.getJointState(robot, i)

            state.append(joint[0])
            state.append(joint[1])

        # ОРИЕНТАЦИЯ

        _, orientation = p.getBasePositionAndOrientation(robot)

        roll, pitch, yaw = p.getEulerFromQuaternion(orientation)

        state.extend([roll,pitch,yaw])

        # ЦЕЛЕВЫЕ ПАРАМЕТРЫ

        state.append(self.target_speed)

        angle_error = self.normalize_angle(self.target_angle - yaw)
        state.append(angle_error)

        state_arr = np.asarray(state,dtype=np.float64)

        # СТАТИСТИКА

        if update_stats:
            self.update_state_stats(state_arr)

        # НОРМАЛИЗАЦИЯ

        normalized = (state_arr - self.state_mean) / (self.state_std + self.eps)

        # Дополнительная защита
        normalized = np.clip(normalized,-10.0,10.0)

        return normalized.astype(np.float32).tolist()

    def get_state_for_agent(self, robot, agent):

        state = []

        for i in range(18):
            joint = p.getJointState(robot,i)

            state.append(joint[0])
            state.append(joint[1])

        _, orientation = p.getBasePositionAndOrientation(robot)

        roll, pitch, yaw = p.getEulerFromQuaternion(orientation)

        state.extend([roll,pitch,yaw])

        state.append(self.target_speed)

        angle_error = self.normalize_angle(self.target_angle - yaw)

        state.append(angle_error)

        state_arr = np.asarray(state,dtype=np.float64)

        normalized = (state_arr - self.state_mean) / (self.state_std + self.eps)

        normalized = np.clip(normalized,-10.0,10.0)

        return normalized.astype(np.float32).tolist()

    def reset_robot(self, robot, position):
        x, y = position
        noise_x = random.uniform(-0.05, 0.05)
        noise_y = random.uniform(-0.05, 0.05)

        p.resetBasePositionAndOrientation(
            robot,
            [x + noise_x, y + noise_y, 0.3],
            [0, 0, 0, 1]
        )
        p.resetBaseVelocity(robot, [0, 0, 0], [0, 0, 0])

        for i in range(18):
            p.resetJointState(robot, i, 0, 0)

    def step(self, robots, actions):

        # ПРИМЕНЯЕМ ДЕЙСТВИЯ

        for i, robot in enumerate(robots):
            self.apply_action(robot,actions[i])

        # СИМУЛЯЦИЯ

        for _ in range(self.action_repeat):
            p.stepSimulation()

        rewards = []
        dones = []
        states = []

        # СОБИРАЕМ РЕЗУЛЬТАТ

        for robot in robots:
            r, d = self.get_reward(robot)

            rewards.append(r)
            dones.append(d)

            states.append(self.get_state(robot,update_stats=False))

        return states, rewards, dones

    def apply_action(self, robot, action):

        for i in range(18):
            current = p.getJointState(robot,i)[0]

            delta = float(action[i]) * ACTION_SCALE

            target = current + delta

            # Защита от ухода сустава в ненормальное положение
            target = np.clip(target,-1.5,1.5)

            p.setJointMotorControl2(robot,i,p.POSITION_CONTROL,targetPosition=float(target),force=40)

    def get_reward(self, robot):
        reward = 1.0
        dead = False

        velocity, angular_velocity = p.getBaseVelocity(robot)

        vx, vy, vz = velocity
        wx, wy, wz = angular_velocity

        position, orientation = p.getBasePositionAndOrientation(robot)
        roll, pitch, yaw = p.getEulerFromQuaternion(orientation)

        height = position[2]
        target_height = 0.20

        # 1. ВЫСОТА

        height_error = height - target_height
        reward -= height_error * height_error * 60.0


        # 2. НАКЛОН

        tilt_error = roll ** 2 + pitch ** 2
        reward -= tilt_error * 3.0

        # 3. ДВИЖЕНИЕ КОРПУСА

        horizontal_speed = math.sqrt(vx * vx + vy * vy)
        reward -= horizontal_speed * 1.0

        # 4. ВРАЩЕНИЕ

        angular_speed = math.sqrt(wx * wx + wy * wy)
        reward -= angular_speed * 0.3

        # КАЧЕСТВО СТОЙКИ

        stability_score = 1.0

        stability_score -= min(abs(roll) / 0.2, 1.0)
        stability_score -= min(abs(pitch) / 0.2, 1.0)
        stability_score -= min(abs(height_error) / 0.06, 1.0)
        stability_score -= min(horizontal_speed / 0.3, 1.0)

        stability_score = max(0.0, stability_score)

        reward += stability_score * 10.0

        # 6. КОНТАКТ ВСЕХ НОГ
        FOOT_LINKS = [2, 5, 8, 11, 14, 17]
        foot_contacts = 0

        for link in FOOT_LINKS:
            contacts = p.getContactPoints(bodyA=robot,bodyB=self.plane,linkIndexA=link)
            if contacts:
                foot_contacts += 1

        reward += foot_contacts * 1.0

        # 7. КОНТАКТ КОРПУСА

        contacts = p.getContactPoints(bodyA=robot,bodyB=self.plane)
        body_contact = False

        for contact in contacts:
            link_index = contact[3]
            normal_force = contact[9]
            if link_index == -1 and normal_force > 0:
                body_contact = True
                break

        # 8. ПАДЕНИЕ

        if abs(roll) > 1.0 or abs(pitch) > 1.0:
            dead = True
            reward -= 50.0

        # Пока НЕ убиваем за body_contact
        # После диагностики можно добавить:
        #
        # if body_contact:
        #     dead = True
        #     reward -= 10.0

        # =========================
        # 9. NaN / Inf
        # =========================

        if math.isnan(reward) or math.isinf(reward):
            reward = -10.0
            dead = True

        return reward, dead


# АГЕНТ (PPO)
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 1))
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self._initialize_weights()

        self.actor.to(DEVICE)
        self.critic.to(DEVICE)
        self.log_std.data = self.log_std.data.to(DEVICE)

        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()) + [self.log_std],lr=LEARNING_RATE)

    def _initialize_weights(self):
        def init(layer):
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        self.actor.apply(init)
        self.critic.apply(init)

    def get_action(self, state, deterministic=False):
        state_t = torch.tensor(state,dtype=torch.float32,device=DEVICE).unsqueeze(0)

        mean = self.actor(state_t)

        std = torch.exp(torch.clamp(self.log_std, -2, 1)).unsqueeze(0)

        dist = Normal(mean, std)

        if deterministic:
            action_norm = mean
        else:
            action_norm = dist.sample()

        action_norm = torch.clamp(action_norm, -1.0, 1.0)

        log_prob = dist.log_prob(action_norm).sum(dim=-1)
        value = self.critic(state_t).squeeze(-1)

        return action_norm.squeeze(0).detach().cpu().numpy(),log_prob.item(),value.item()

    def train_ppo(self, states, actions, old_log_probs, returns, advantages):
        # --- ЖЁСТКАЯ ПРОВЕРКА И ОБРЕЗКА ДЛИН ---
        lens = [len(states), len(actions), len(old_log_probs), len(returns), len(advantages)]
        min_len = min(lens)
        if min_len == 0:
            print("PPO: Нет данных для обучения, пропускаем шаг.")
            return

        if any(l != min_len for l in lens):
            print(f"PPO: Длины буферов не совпадают: {lens}. Обрезаем до {min_len}.")
            states = states[:min_len]
            actions = actions[:min_len]
            old_log_probs = old_log_probs[:min_len]
            returns = returns[:min_len]
            advantages = advantages[:min_len]

        # Конвертируем в тензоры
        states_t = torch.from_numpy(np.array(states, dtype=np.float32)).to(DEVICE)
        actions_t = torch.from_numpy(np.array(actions, dtype=np.float32)).to(DEVICE)
        old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32, device=DEVICE)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=DEVICE).unsqueeze(1)

        n_samples = states_t.shape[0]
        adv_mean = advantages_t.mean()
        adv_std = advantages_t.std() + 1e-8
        advantages_norm = (advantages_t - adv_mean) / adv_std

        total_loss = 0.0
        num_batches = 0

        for epoch in range(PPO_EPOCHS):
            indices = list(range(n_samples))
            random.shuffle(indices)
            for start in range(0, n_samples, MINIBATCH_SIZE):
                end = min(start + MINIBATCH_SIZE, n_samples)
                batch_idx = indices[start:end]  # обычный список int

                states_batch = states_t[batch_idx]
                actions_batch = actions_t[batch_idx]
                old_log_probs_batch = old_log_probs_t[batch_idx]
                returns_batch = returns_t[batch_idx]
                adv_batch = advantages_norm[batch_idx]

                new_log_probs, entropy, values = self.evaluate_batch(states_batch, actions_batch)

                if torch.isnan(new_log_probs).any():
                    continue

                ratio = torch.exp(new_log_probs - old_log_probs_batch)
                ratio = torch.clamp(ratio, 0.0, 10.0)

                surr1 = ratio * adv_batch.squeeze(1)
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * adv_batch.squeeze(1)

                actor_loss = -torch.min(surr1, surr2).mean()
                value_loss = (returns_batch.squeeze(1) - values).pow(2).mean()
                entropy_loss = entropy.mean()

                loss = actor_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()) + [self.log_std],
                    max_norm=0.5
                )
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"PPO Loss: {avg_loss:.4f}")

    def evaluate_batch(self, states_t, actions_t):
        mean = self.actor(states_t)

        std = torch.exp(torch.clamp(self.log_std, -2, 1))

        min_std = 1e-6
        std = torch.max(std, torch.full_like(std, min_std))

        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions_t).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = self.critic(states_t).squeeze(-1)
        return log_probs, entropy, values

    def copy_from(self, other):

        self.actor.load_state_dict(copy.deepcopy(other.actor.state_dict()))

        self.critic.load_state_dict(copy.deepcopy(other.critic.state_dict()))

        with torch.no_grad():
            self.log_std.copy_(other.log_std.detach().clone())

        # Новый агент получает чистый optimizer
        self.optimizer = optim.Adam(list(self.actor.parameters()) +list(self.critic.parameters()) +[self.log_std],lr=LEARNING_RATE)

    def mutate(self,actor_strength=0.001,critic_strength=CRITIC_MUTATION,log_std_strength=LOG_STD_MUTATION):

        with torch.no_grad():

            # Actor
            for param in self.actor.parameters():
                noise = torch.randn_like(param) * actor_strength
                param.add_(noise)

            # Critic
            for param in self.critic.parameters():
                noise = torch.randn_like(param) * critic_strength
                param.add_(noise)

            # Стандартное отклонение политики
            noise = (torch.randn_like(self.log_std) * log_std_strength)

            self.log_std.add_(noise)

            # Защита от слишком большой/маленькой случайности
            self.log_std.clamp_(-2.0, 1.0)

def clone_agent(agent):
    clone = PPOAgent(STATE_DIM,ACTION_DIM)

    clone.copy_from(agent)

    return clone

def calculate_gae(rewards, values, dones, last_value):

    advantages = []
    gae = 0.0

    values = list(values) + [last_value]

    for t in reversed(range(len(rewards))):

        if dones[t]:
            next_value = 0.0
        else:
            next_value = values[t + 1]

        delta = (rewards[t] + GAMMA * next_value - values[t])

        gae = (delta + GAMMA * GAE_LAMBDA * gae * (1.0 - float(dones[t])))

        advantages.insert(0,gae)

    returns = [adv + val for adv, val in zip(advantages,values[:-1])]

    advantages = np.asarray(advantages,dtype=np.float32)

    returns = np.asarray(returns,dtype=np.float32)

    # Защита PPO от NaN / Inf
    advantages = np.nan_to_num(advantages,nan=0.0,posinf=10.0,neginf=-10.0)

    returns = np.nan_to_num(returns,nan=0.0,posinf=10.0,neginf=-10.0)

    return advantages.tolist(),returns.tolist()


class RobotAgent:
    """Обёртка, чтобы хранить буфер и статистику для одного робота"""
    def __init__(self, model):
        self.model = model
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.episode_rewards = []   

def create_robots(num_agents):
    robots = []
    positions = []
    robot_urdf = "hexapod.urdf"


    for i in range(num_agents):
        x = 0.0
        y = i * 1.2  # увеличим дистанцию про запас
        robot = p.loadURDF(robot_urdf, [x, y, 0.3], useFixedBase=False)
        FOOT_LINKS = [2, 5, 8, 11, 14, 17]
        for link_idx in FOOT_LINKS:
            p.changeDynamics(robot, link_idx, lateralFriction=3.0, spinningFriction=1.5)
        robots.append(robot)
        positions.append([x, y])

        # Отключаем коллизии между всеми парами роботов
        for j in range(i):
            p.setCollisionFilterPair(robot, robots[j], -1, -1, enableCollision=False)

    return robots, positions

def save_model(agent, world, path):
    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory,exist_ok=True)

    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'log_std': agent.log_std.detach().cpu(),
        'optimizer_state_dict': agent.optimizer.state_dict(),

        'state_mean': world.state_mean,
        'state_std': world.state_std,
        'state_M2': world.state_M2,
        'state_count': world.count,
        'normalization_frozen': world.normalize_frozen
    }, path)

    print(f"Model saved to {path}")

def load_model(agent, world, path):

    if not os.path.exists(path):

        print(f"Файл не найден: {path}")
        return False

    checkpoint = torch.load(path,map_location=DEVICE,weights_only=False)

    # PPO
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.critic.load_state_dict(checkpoint['critic_state_dict'])

    with torch.no_grad():
        agent.log_std.copy_(checkpoint['log_std'].to(DEVICE))

    if 'optimizer_state_dict' in checkpoint:
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # ОБЩАЯ НОРМАЛИЗАЦИЯ
    if 'state_mean' in checkpoint:
        world.state_mean = np.asarray(checkpoint['state_mean'],dtype=np.float64)

    if 'state_std' in checkpoint:
        world.state_std = np.asarray(checkpoint['state_std'],dtype=np.float64)

    if 'state_M2' in checkpoint:
        world.state_M2 = np.asarray(checkpoint['state_M2'],dtype=np.float64)

    if 'state_count' in checkpoint:
        world.count = int(checkpoint['state_count'])

    if checkpoint.get('normalization_frozen',False):
        world.freeze_normalization()

    print(f"Model loaded from {path}")

    return True

def create_test_robot():

    robot_urdf = "hexapod.urdf"

    test_robot = p.loadURDF(robot_urdf,[100, 100, 0.3],useFixedBase=False)

    print("TEST ROBOT JOINTS:",p.getNumJoints(test_robot))

    FOOT_LINKS = [2, 5, 8, 11, 14, 17]

    for link_idx in FOOT_LINKS:
        p.changeDynamics(test_robot,link_idx,lateralFriction=3.0,spinningFriction=1.5)

    return test_robot

def reset_test_robot(test_robot):
    p.resetBasePositionAndOrientation(test_robot,
        [100.0, 100.0, 0.3],
        [0, 0, 0, 1])

    p.resetBaseVelocity(test_robot,
        [0, 0, 0],
        [0, 0, 0])

    num_joints = p.getNumJoints(test_robot)

    for i in range(num_joints):
        p.resetJointState(test_robot,i,targetValue=0.0,targetVelocity=0.0)

    # Обновляем состояние столкновений после reset
    p.performCollisionDetection()

def evaluate_agent_once(world,agent,test_robot,max_steps=2048,num_tests=5):

    if test_robot is None:
        return -1e6

    test_scores = []

    for test in range(num_tests):
        reset_test_robot(test_robot)
        total_reward = 0.0

        for step in range(max_steps):
            state = world.get_state_for_agent(test_robot,agent)
            action, _, _ = agent.get_action(state,deterministic=True)
            _, rewards, dones = world.step([test_robot],[action])
            reward = rewards[0]

            if not math.isfinite(reward):
                reward = -10.0

            total_reward += reward

            if dones[0]:
                break

        test_scores.append(total_reward)

    mean_score = float(np.min(test_scores))

    print(f"   Тесты: "f"{[round(x, 2) for x in test_scores]} "f"→ Минимальное: {mean_score:.2f}")

    return mean_score

def evolve_population(world,agents,test_robot,num_leaders=2,global_best_agent=None,global_best_score=-float('inf'),mutation_strength=0.001):

    print("\n" + "=" * 60)
    print("ЭВОЛЮЦИЯ ПОПУЛЯЦИИ")
    print("=" * 60)

    # 1. ОЦЕНКА ВСЕХ АГЕНТОВ

    scores = []

    for i, agent in enumerate(agents):
        score = evaluate_agent_once(world,agent,test_robot,max_steps=MAX_EPISODE_STEPS,num_tests=3)
        scores.append(score)
        print(
            f"Robot {i}: "
            f"{score:.2f}"
        )

    # 2. ОПРЕДЕЛЯЕМ ЛУЧШЕГО

    ranking = np.argsort(scores)[::-1]

    best_idx = int(ranking[0])
    best_score = scores[best_idx]

    print("\nРЕЙТИНГ:")

    for rank, idx in enumerate(ranking):
        print(f"{rank + 1}. "f"Robot {idx} = "f"{scores[idx]:.2f}")

    # 3. ОБНОВЛЯЕМ GLOBAL BEST

    if global_best_agent is None or best_score > global_best_score:

        global_best_agent = clone_agent(agents[best_idx])
        global_best_score = best_score
        save_model(global_best_agent,world,os.path.join(CHECKPOINT_DIR,"hexapod_ppo_best.pth"))

        print(f"\nНОВЫЙ GLOBAL BEST: "f"{global_best_score:.2f}")

    else:
        print(f"\nGLOBAL BEST СОХРАНЁН: "f"{global_best_score:.2f}")

    # 4. ТЕКУЩИЕ ЛИДЕРЫ

    leaders_indices = ranking[:num_leaders]

    print(f"\nЛИДЕРЫ: "f"{leaders_indices.tolist()}")

    # 5. КОПИРУЕМ ТЕКУЩИХ ЛИДЕРОВ

    leader_models = []

    for idx in leaders_indices:
        leader = clone_agent(agents[idx])
        leader_models.append(leader)

    # 6. НОВОЕ ПОКОЛЕНИЕ

    new_agents = []

    # GLOBAL ELITE

    elite = clone_agent(global_best_agent)
    new_agents.append(elite)
    print(f"\n🛡 GLOBAL ELITE: "f"{global_best_score:.2f}")

    # ВТОРОЙ ЛИДЕР

    if num_leaders > 1:
        second_elite = clone_agent(leader_models[0])

        # если текущий лучший и global best
        # совпадают, используем второго лидера
        if best_score == global_best_score and len(leader_models) > 1:
            second_elite = clone_agent(leader_models[1])
        new_agents.append(second_elite)

    # ПОТОМКИ

    while len(new_agents) < len(agents):

        parent = leader_models[(len(new_agents) - num_leaders) % len(leader_models)]
        child = clone_agent(parent)
        child.mutate(actor_strength=mutation_strength,critic_strength=CRITIC_MUTATION,log_std_strength=LOG_STD_MUTATION)
        new_agents.append(child)

    # ПРОВЕРКА

    print(f"\nНовое поколение: "f"{len(new_agents)} агентов")

    print(f"Global Elite: 1")

    print(f"Мутантов: "f"{len(new_agents) - 2}")

    return new_agents,scores,global_best_agent,global_best_score

def update_mutation(current_mutation,improved,stagnant_generations):
    """
    Адаптивное изменение силы мутации.

    improved = True:
        global best улучшился -> уменьшаем мутацию

    improved = False:
        прогресса нет -> после нескольких поколений
        увеличиваем мутацию
    """

    if improved:
        # Мы нашли новое хорошее решение.
        # Делаем поиск более точным.
        current_mutation *= MUTATION_DECREASE_FACTOR
        stagnant_generations = 0
        print(f"Улучшение! "f"Мутация уменьшена до {current_mutation:.6f}")

    else:
        stagnant_generations += 1

        # Если несколько поколений нет прогресса,
        # усиливаем исследование пространства параметров.
        if stagnant_generations >= STAGNATION_LIMIT:
            current_mutation *= MUTATION_INCREASE_FACTOR
            stagnant_generations = 0
            print(f"ЗАСТОЙ! "f"Мутация увеличена до {current_mutation:.6f}")

    # Жёсткие границы
    current_mutation = np.clip(current_mutation,MIN_ACTOR_MUTATION,MAX_ACTOR_MUTATION)

    return current_mutation, stagnant_generations


def train():
    # Начальная мутация
    now_mutation = 0.005

    # Сколько поколений подряд не было улучшения
    stagnant_generations = 0

    next_evolution = EVOLUTION_EVERY_STEPS

    num_agents = 10

    world = World()  # без робота внутри — создадим сами
    robots, positions = create_robots(num_agents)

    test_robot = create_test_robot()

    if test_robot is None:
        print("Не удалось создать тестового робота.")
        return

    # У каждого робота СВОЯ нейросеть
    agents = [PPOAgent(STATE_DIM, ACTION_DIM) for _ in range(num_agents)]

    if MODE == "TEST":
        if not load_model(agents[0], MODEL_FILE):
            print("Для режима TEST не удалось загрузить модель. Завершаем.")
            return
        print("Запуск в режиме TEST — обучение отключено.")
        # Здесь можно добавить отдельный цикл теста, если нужно
        return

    total_steps = 0

    global_best_agent = None
    global_best_score = -float('inf')

    model_loaded = False

    for agent in agents:

        loaded = load_model(agent,world,MODEL_FILE)

        if loaded:
            model_loaded = True

    if model_loaded:
        # Восстанавливаем Global Best
        global_best_agent = clone_agent(agents[0])

        print("Global Best загружен")

    print(f"Начало обучения: {num_agents} роботов, у каждого своя сеть.")

    episode_step_count = [0] * num_agents

    while total_steps < TOTAL_STEPS_TO_TRAIN:

        # --- СБОР ДАННЫХ (ROLLOUT) ---
        states_buf = []
        actions_buf = []
        log_probs_buf = []
        values_buf = []
        rewards_buf = []
        dones_buf = []
        if not world.normalize_frozen and total_steps >= NORMALIZATION_WARMUP_STEPS:
            world.freeze_normalization()

        for _ in range(ROLLOUT_STEPS):
            batch_states = []
            batch_actions = []
            batch_log_probs = []
            batch_values = []

            # 1) Получение состояния и действия
            for i, robot in enumerate(robots):
                # Сначала обновляем общую статистику
                world.get_state(robot,update_stats=True)

                # Затем используем snapshot статистики конкретного агента
                state = world.get_state_for_agent(robot,agents[i])
                action, log_prob, value = agents[i].get_action(state)

                batch_states.append(state)
                batch_actions.append(action)
                batch_log_probs.append(log_prob)
                batch_values.append(value)

            states_buf.append(batch_states)
            actions_buf.append(batch_actions)
            log_probs_buf.append(batch_log_probs)
            values_buf.append(batch_values)

            # Шаг симуляции
            _, rewards, dones = world.step(robots, batch_actions)

            rewards_buf.append(rewards)
            dones_buf.append(dones)
            total_steps += len(robots)

            # 2) Сброс: по смерти ИЛИ по лимиту шагов
            for i, d in enumerate(dones):
                # Увеличиваем счётчик шагов эпизода
                episode_step_count[i] += 1

                # Сброс, если умер ИЛИ если достигли лимита шагов
                if d or episode_step_count[i] >= MAX_EPISODE_STEPS:
                    world.reset_robot(robots[i], positions[i])
                    # Сбрасываем счётчик для этого робота
                    episode_step_count[i] = 0


        # --- ОБУЧЕНИЕ (для каждого агента отдельно) ---
        for i in range(num_agents):

            s_i = [states[i] for states in states_buf]
            a_i = [actions[i] for actions in actions_buf]
            lp_i = [logs[i] for logs in log_probs_buf]
            v_i = [values[i] for values in values_buf]
            r_i = [rewards[i] for rewards in rewards_buf]
            d_i = [dones[i] for dones in dones_buf]

            # GAE
            if d_i[-1]:
                last_val = 0.0
            else:
                last_val = v_i[-1]

            advantages, returns = calculate_gae(r_i,v_i,d_i,last_val)

            agents[i].train_ppo(s_i, a_i, lp_i, returns, advantages)

        # --- СТАТИСТИКА И СОХРАНЕНИЕ ---
        # Средняя суммарная награда по всем роботам за этот rollout
        episode_reward_per_agent = [sum(r_list) for r_list in zip(*rewards_buf)]
        avg_episode_reward = np.mean(episode_reward_per_agent)

        if total_steps >= SAVE_EVERY_STEPS and (total_steps // SAVE_EVERY_STEPS != (total_steps - num_agents * ROLLOUT_STEPS) // SAVE_EVERY_STEPS):
            save_model(agents[0],world,os.path.join(CHECKPOINT_DIR,f"hexapod_ppo_step_{total_steps}.pth"))

        # --- ЭВОЛЮЦИЯ (выбираем лучшего кандидата и копируем всем) ---
        if USE_MUTATION_BASED_SELECTION:

            # ЭВОЛЮЦИЯ

            if total_steps >= next_evolution:

                next_evolution += EVOLUTION_EVERY_STEPS

                # Запоминаем старый GLOBAL BEST

                old_global_best = global_best_score

                # ЭВОЛЮЦИЯ

                agents, scores, global_best_agent, global_best_score = evolve_population(world,agents,test_robot,num_leaders=NUM_LEADERS,global_best_agent=global_best_agent,global_best_score=global_best_score,mutation_strength=now_mutation)

                # Определяем, был ли прогресс

                improved = global_best_score > old_global_best

                # Для самого первого поколения:
                if old_global_best == -float('inf'):
                    improved = True

                # АДАПТИВНАЯ МУТАЦИЯ

                now_mutation, stagnant_generations = update_mutation(now_mutation,improved,stagnant_generations)

                print(f"Mutation: {now_mutation:.6f} | "f"Stagnation: {stagnant_generations}")

                # Global Elite
                agents[0] = clone_agent(global_best_agent)

                # ПРОВЕРКА ELITE

                elite_score = evaluate_agent_once(world,agents[0],test_robot,max_steps=MAX_EPISODE_STEPS,num_tests=2)

                print(f"ПРОВЕРКА ELITE: "f"{elite_score:.2f} | "f"GLOBAL BEST: {global_best_score:.2f}")

                # СБРОС ПОСЛЕ ЭВОЛЮЦИИ

                for i in range(num_agents):
                    world.reset_robot(robots[i],positions[i])
                    episode_step_count[i] = 0

        avg_reward = np.mean([r for r_list in rewards_buf for r in r_list])
        print(f"Steps: {total_steps} | Avg reward: {avg_reward:.3f} | Best avg episode reward: {global_best_score:.2f}")


    print("Обучение завершено.")


    if test_robot is not None:
        p.removeBody(test_robot)

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\nОбучение остановлено пользователем.")
    finally:
        p.disconnect()
