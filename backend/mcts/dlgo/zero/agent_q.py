import numpy as np
from keras import regularizers
from tensorflow import keras
from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from ..agent import Agent

__all__ = [
    'ZeroAgent',
]

from ..goboard import Move

# tag::branch_struct[]
from ..gotypes import Player
from ..scoring import compute_game_result


class Branch:
    def __init__(self, prior):
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0


# end::branch_struct[]


# tag::node_class_defn[]
class ZeroTreeNode:
    def __init__(self, state, value, priors, parent, last_move):
        self.state = state
        self.value = value
        self.parent = parent
        self.last_move = last_move
        self.total_visit_count = 1
        self.branches = {}
        for move, p in priors.items():
            if state.is_valid_move(move):
                self.branches[move] = Branch(p)
        self.children = {}

        # 初始化每个合法动作的Q值和E值
        self.Q = {}
        self.E = {}
        for move in state.legal_moves():
            self.Q[move] = 0.0  # 初始化Q值为0
            self.E[move] = 0.0  # 初始化E值为0

    def moves(self):  # <3>
        return self.branches.keys()  # <3>

    def add_child(self, move, child_node):  # <4>
        self.children[move] = child_node  # <4>

    def has_child(self, move):  # <5>
        return move in self.children  # <5>

    def get_child(self, move):  # <6>
        return self.children[move]  # <6>

    # end::node_class_body[]

    # tag::node_record_visit[]
    def record_visit(self, move, value):
        # 如果是新动作，初始化其Q值
        if move not in self.Q:
            self.Q[move] = 0.0  # 或根据您的经验选择一个初始值
            print(f"Initializing Q value for new move: {move}")

        # 更新访问次数和总值
        self.total_visit_count += 1
        if move in self.branches:
            self.branches[move].visit_count += 1
            self.branches[move].total_value += value

            # print(f"Recording visit for move: {move}, new visit count: {self.branches[move].visit_count}")  # 输出访问计数的更新
            # with open("training_log.txt", "a") as file:
            #     file.write(f"Recording visit for move: {move}, new visit count: {self.branches[move].visit_count}")  # 输出访问计数的更新
        else:
            # 处理其他未知情况
            print("Unknown move type or move not found in branches:", move)

    # end::node_record_visit[]

    # tag::node_class_helpers[]
    def expected_value(self, move):
        branch = self.branches[move]
        if branch.visit_count == 0:
            return 0.0
        return branch.total_value / branch.visit_count

    def prior(self, move):
        print(self.branches[move].prior)
        return self.branches[move].prior

    def visit_count(self, move):
        if move in self.branches:
            return self.branches[move].visit_count
        return 0


# end::node_class_helpers[]


# tag::zero_defn[]
class ZeroAgent(Agent):
    # end::zero_defn[]
    def __init__(self, model, encoder, rounds_per_move=1600, c=2.0):
        self.model = model
        self.encoder = encoder
        self.collector = None
        self.num_rounds = rounds_per_move
        self.c = c

    def initialize_game(self, initial_state):
        self.root = self.create_node(initial_state)

    # 根据当前状态和动作计算即时奖励 R(s,a)
    def calculate_R_s_a(self, game_state, move):
        # 判断游戏是否结束
        if game_state.is_over():
            game_result = compute_game_result(game_state)
            current_player = game_state.next_player.other  # 获取当前玩家
            if game_result.winner == current_player:
                return 1  # 赢得比赛
            elif game_result.winner is None:
                return 0  # 和棋
            else:
                return -1  # 输掉比赛
        else:
            # 使用 compute_game_result 计算目前的游戏结果
            game_result = compute_game_result(game_state)
            black_score = game_result.b
            white_score = game_result.w

            # 计算目数差和吃子数差
            black_captures = game_state.board.num_captures(Player.black)
            white_captures = game_state.board.num_captures(Player.white)

            V_count = black_score - white_score
            V_capture = black_captures - white_captures

            return V_count + V_capture

    # Sarsa(λ) 更新逻辑
    def update_with_sarsa_lambda(self, node, gamma, lambda_):
        while node is not None and node.parent is not None:
            next_node = node.parent
            next_move = next_node.last_move

            # 检查 next_move 是否有效
            if next_move is not None and next_move in next_node.E and next_move in next_node.Q:
                R_s_a = self.calculate_R_s_a(node.state, next_move)
                next_q = next_node.Q[next_move]
                node.E[next_move] = R_s_a + gamma * next_q - node.Q[next_move]
                node.Q[next_move] += lambda_ * node.E[next_move]

            node = next_node

    # Q-Learning 更新逻辑
    def update_with_q_learning(self, node, gamma):
        while node is not None:

            move = node.last_move
            if move is not None and move in node.Q and node.has_child(move):
                try:
                    # 尝试提取所有子节点的Q值
                    child_q_values = [child_node.Q.get(child_move, 0) for child_node in
                                      node.children[move].children.values() for child_move in child_node.Q]
                    best_next_q = max(child_q_values) if child_q_values else 0
                except KeyError:
                    # 处理可能的KeyError异常
                    best_next_q = 0

                # 更新当前动作的Q值
                node.Q[move] += gamma * (best_next_q - node.Q.get(move, 0))
            if node.parent is None and node.state.is_over():  # 检查是否为根节点且游戏结束
                with open("training_log.txt", "a") as file:
                    file.write(f"Final Q values: {node.Q}\n")
            node = node.parent

    # tag::zero_select_move_defn[]
    def select_move(self, game_state):
        root = self.create_node(game_state)
        for i in range(self.num_rounds):
            node = root
            next_move = self.select_branch(node)

            while node.has_child(next_move):
                node = node.get_child(next_move)
                next_move = self.select_branch(node)

            # 检查 next_move 是否为 None
            if next_move is None:
                # 没有有效的移动，选择一个随机动作或跳过这个回合的更新
                valid_moves = list(node.moves())
                if valid_moves:
                    next_move = np.random.choice(valid_moves)
                    new_state = node.state.apply_move(next_move)
                    child_node = self.create_node(new_state, move=next_move, parent=node)
                else:
                    # 没有可用的动作，跳过这个回合的更新
                    continue
            else:
                new_state = node.state.apply_move(next_move)
                child_node = self.create_node(new_state, move=next_move, parent=node)

            # 在这里记录访问
            if next_move in node.Q:
                node.record_visit(next_move, node.Q[next_move])
            else:
                # 如果next_move不在Q中，记录访问并初始化Q值
                print(f"Warning: Move {next_move} not found in Q values, initializing.")
                node.Q[next_move] = 0  # 初始化Q值
                node.record_visit(next_move, 0)  # 记录访问并使用初始值
            # 反向更新节点
            self.update_with_sarsa_lambda(child_node, gamma=0.9, lambda_=0.5)
            self.update_with_q_learning(child_node, gamma=0.9)
            # 经验收集
            if self.collector is not None:
                root_state_tensor = self.encoder.encode(game_state)
                visit_counts = np.array([
                    root.visit_count(self.encoder.decode_move_index(idx))
                    for idx in range(self.encoder.num_moves())
                ])
                visit_sums = np.sum(visit_counts)

                # 可能需要将此逻辑放在collector中或直接在这里处理
                # 记录visit_counts和visit_sums
                with open("training_log.txt", "a") as file:
                    file.write(f"visit_counts:{visit_counts.tolist()}, visit_sums:{visit_sums}\n")

            self.collector.record_decision(root_state_tensor, visit_counts)

        # 选择访问次数最多的动作作为此次动作
        return max(root.moves(), key=root.visit_count)

    # end::zero_select_max_visit_count[]

    def set_collector(self, collector):
        self.collector = collector

    # tag::zero_select_branch[]
    def select_branch(self, node, c_puct=1.0):
        best_score = -float('inf')
        best_move = None

        total_n = sum(branch.visit_count for branch in node.branches.values())
        if total_n == 0:
            return np.random.choice(list(node.moves()))  # 如果没有访问过，随机选择一个动作

        for move, branch in node.branches.items():
            Q = node.Q.get(move, 0)
            P = branch.prior
            N = branch.visit_count
            U = c_puct * P * np.sqrt((2 * np.log(total_n + 1e-10) / (N + 1)))  # 避免除以零
            score = Q + U

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

        # end::zero_select_branch[]

    # tag::zero_create_node[]
    def create_node(self, game_state, move=None, parent=None):
        state_tensor = self.encoder.encode(game_state)
        model_input = np.array([state_tensor])  # <1>
        priors, values = self.model.predict(model_input)
        priors = priors[0]  # <2>
        value = values[0][0]  # <2>
        move_priors = {  # <3>
            self.encoder.decode_move_index(idx): p  # <3>
            for idx, p in enumerate(priors)  # <3>
        }  # <3>
        new_node = ZeroTreeNode(
            game_state, value,
            move_priors,
            parent, move)
        if parent is not None:
            parent.add_child(move, new_node)
        return new_node

    # end::zero_create_node[]

    # tag::zero_train[]
    def train(self, experience, learning_rate, batch_size):
        num_examples = experience.states.shape[0]

        if num_examples == 0:
            print("Experience data is empty. Skipping training.")
            return

        model_input = experience.states
        visit_sums = np.sum(experience.visit_counts, axis=1).reshape((-1, 1))
        with open("training_log.txt", "a") as file:
            file.write(f"visit_counts:{experience.visit_counts},Visit_sums:{visit_sums}\n")
        visit_sums = np.where(visit_sums == 0, 1e-10, visit_sums)  # 避免除以零

        action_probs = experience.visit_counts / visit_sums  # π_i，动作概率
        value_targets = experience.rewards.reshape((-1, 1))  # z_i，值目标

        # 定义自定义损失函数
        def custom_loss(y_true, y_pred):
            y_pred_policy = y_pred[0]  # 策略输出
            y_pred_value = y_pred[1]  # 价值输出


            y_true_policy = y_true[:, :-1]  # 所有除了最后一个元素之外
            y_true_value = y_true[:, -1:]  # 只有最后一个元素

            print(y_true_policy)
            print(y_true_value)
            print(y_pred_policy)
            print(y_pred_value)
            # 确保y_true和y_pred的数据类型一致
            y_true_policy = tf.cast(y_true_policy, tf.float32)
            y_true_value = tf.cast(y_true_value, tf.float32)
            y_pred_policy = tf.cast(y_pred_policy, tf.float32)
            y_pred_value = tf.cast(y_pred_value, tf.float32)

            # 计算损失
            policy_loss = tf.reduce_mean(
                tf.reduce_sum(-y_true_policy * tf.math.log(tf.clip_by_value(y_pred_policy, 1e-10, 1)), axis=1))
            value_loss = tf.reduce_mean(tf.square(y_true_value - y_pred_value))
            total_loss = policy_loss + value_loss

            return total_loss

        # 合并值目标和动作概率为y_true
        y_true = tf.concat([value_targets, action_probs], axis=1)

        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss=custom_loss)
        history = self.model.fit(model_input, y_true, batch_size=batch_size, epochs=1)

        with open("training_log.txt", "a") as file:
            file.write(f"Visit:{visit_sums},Loss: {history.history['loss'][-1]}, Reward: {np.mean(value_targets)}\n")
# end::zero_train[]
