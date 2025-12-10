import random

import numpy as np
from keras.optimizers import SGD, Adam
from tensorflow.python.keras.losses import MeanSquaredError

from ..agent import Agent
import tensorflow as tf

__all__ = [
    'ZeroAgent',
]

# tag::branch_struct[]
from ..goboard import Move
from ..gotypes import Player, Point
from ..utils import print_move


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

        # 新增属性
        self.E = {move: 0.0 for move in self.branches}  # 初始化每个动作的E值

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
        if move not in self.branches:
            # 如果move不在branches中，则初始化它
            self.branches[move] = Branch(0.0)  # 使用适当的初始值
        self.total_visit_count += 1
        self.branches[move].visit_count += 1
        self.branches[move].total_value += value

    # end::node_record_visit[]

    # tag::node_class_helpers[]
    def expected_value(self, move):
        branch = self.branches[move]
        if branch.visit_count == 0:
            return 0.0
        return branch.total_value / branch.visit_count

    def prior(self, move):
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

    def calculate_normal_distribution_value(self, x, y, sigma=2, mu=6.5):
        """
                        计算给定坐标(x, y)在二维正态分布下的值。
                        参数:
                        - x, y: 棋盘上的坐标，范围为[0, 13]。
                        - sigma: 正态分布的标准差，这里假设x和y方向相同。
                        - mu: 正态分布的均值，这里假设棋盘的中心为(6.5, 6.5)。

                        返回:
                        - 该坐标在二维正态分布下的值。
                        """
        rho = 0  # ρ，x和y的相关系数，在此景中为0，表示x和y是独立的。
        part1 = 1 / (2 * np.pi * sigma * sigma * np.sqrt(1 - rho ** 2))
        part2_exponent = -1 / (2 * (1 - rho ** 2)) * (
                ((x - mu) ** 2) / (sigma ** 2) - 2 * rho * (x - mu) * (y - mu) / (sigma * sigma) + (
                (y - mu) ** 2) / (sigma ** 2))
        value = part1 * np.exp(part2_exponent)
        return value

    def calculate_R_s_a(self, game_state, move):
        if game_state.phase == 1:  # 布局阶段
            x, y = move.point.row - 1, move.point.col - 1  # 调整坐标系以适应正态分布计算
            V_static = self.calculate_normal_distribution_value(x, y)
            return V_static
        else:  # 对战阶段
            V_chain = 7 * len(game_state.find_dalian(game_state.next_player))
            V_gate = 3 * len(game_state.find_triangle(game_state.next_player))
            V_square = 4 * len(game_state.find_squares(game_state.next_player))
            return V_chain + V_gate + V_square

    def update_with_sarsa_lambda(self, node, gamma, lambda_):
        while node is not None and node.parent is not None:
            next_node = node.parent
            next_move = next_node.last_move
            if next_move is not None:
                R_s_a = self.calculate_R_s_a(node.state, next_move)
                next_q = self.get_next_Q(next_node, next_move)
                # 更新E值
                node.E[next_move] = R_s_a + gamma * next_q - node.value
                # 使用E值更新value
                node.value += lambda_ * node.E[next_move]
            node = next_node

    def update_with_q_learning(self, node, gamma):
        while node is not None:
            move = node.last_move
            if move is not None:
                max_next_q = self.get_max_next_Q(node)
                # 直接更新value
                node.value += gamma * (max_next_q - node.value)
            node = node.parent

    def get_next_Q(self, node, next_move):
        """
        获取下一个动作的Q值。
        """
        if next_move in node.children:
            child_node = node.children[next_move]
            return child_node.value
        return 0

    def get_max_next_Q(self, node):
        """
        获取下一个动作的最大Q值。
        """
        if not node.children:
            return 0
        return max(child.value for child in node.children.values())

    def select_move(self, game_state):
        """
        选择下一步动作。
        """
        point1 = Point(game_state.board.num_rows // 2, game_state.board.num_cols // 2)
        point2 = Point(game_state.board.num_rows // 2 + 1, game_state.board.num_cols // 2 + 1)
        if game_state.round == 1 or game_state.round == 2:
            if game_state.board.grid.get(point1) is not None:
                return Move(point=point2)
            elif game_state.board.grid.get(point2) is not None:
                return Move(point=point1)
            else:
                return random.choice([Move(point=point2), Move(point=point1)])

        root = self.create_node(game_state)
        for _ in range(self.num_rounds):
            node = root
            legal_moves = game_state.get_legal_moves()
            if len(legal_moves) == 0:
                return Move.resign()
            # for move in legal_moves:
            #     print_move(game_state.next_player.other, move)
            next_move = self.select_branch(node, legal_moves)
            # print_move(game_state.next_player, next_move)
            while node.has_child(next_move):
                node = node.get_child(next_move)
                next_move = self.select_branch(node)
            new_state = node.state.apply_move(next_move)
            child_node = self.create_node(new_state, next_move, node)

            # 使用反向传播更新节点
            value = -child_node.value
            while node is not None:
                node.record_visit(next_move, value)
                value = -value
                node = node.parent
        # tag::zero_record_collector[]
        if self.collector is not None:
            root_state_tensor = self.encoder.encode(game_state)
            visit_counts = np.array([
                root.visit_count(
                    self.encoder.decode_move_index(idx))
                for idx in range(len(legal_moves))
            ])
            self.collector.record_decision(
                root_state_tensor, visit_counts)
        # end::zero_record_collector[]

        return max(root.moves(), key=root.visit_count)

    def collect_experience(self, node, child_node, game_state):
        root_state_tensor = self.encoder.encode(game_state)
        visit_counts = np.array([
            node.visit_count(self.encoder.decode_move_index(idx))
            for idx in range(self.encoder.num_moves())
        ])
        visit_sums = np.sum(visit_counts)
        self.collector.record_decision(root_state_tensor, visit_counts / visit_sums if visit_sums > 0 else visit_counts)

    def set_collector(self, collector):
        self.collector = collector

    def select_branch(self, node, legal_moves, c_puct=0.5):
        """
        根据UCB公式选择最佳分支，仅考虑合法动作。
        """
        best_move = None
        best_score = -float('inf')
        total_n = sum(branch.visit_count for branch in node.branches.values())
        for move, branch in node.branches.items():
            Q = node.value  # 使用累积长期价值
            P = branch.prior
            N = branch.visit_count
            U = c_puct * P * np.sqrt(total_n) / (1 + N)
            score = Q + U
            if score > best_score:
                # print(score)
                best_score = score
                best_move = move
        if best_move is None and legal_moves:
            # 没有找到最佳移动，但存在合法移动，随机选择一个合法移动
            print("random choice")
            best_move = random.choice(legal_moves)
        return best_move

    def create_node(self, game_state, move=None, parent=None):
        """
        创建新的搜索树节点。
        """
        state_tensor = self.encoder.encode(game_state)
        model_input = np.array([state_tensor])
        priors, values = self.model.predict(model_input)
        value = values[0][0]  # 作为累积长期价值
        priors = np.squeeze(priors)
        # 确保只为合法动作创建分支
        legal_moves = game_state.get_legal_moves()
        move_priors = {move: priors[i] for i, move in enumerate(legal_moves)}
        return ZeroTreeNode(game_state, value, move_priors, parent, move)

    def train(self, experience, learning_rate, batch_size):
        print(f"Training with rewards: {experience.rewards}")
        num_examples = experience.states.shape[0]

        if num_examples == 0:
            print("Experience data is empty. Skipping training.")
            return

        model_input = experience.states

        value_targets = experience.rewards.reshape((-1, 1))  # z_i，值目标

        # 处理 visit_counts
        visit_counts = np.array(experience.visit_counts)
        visit_counts_processed = visit_counts.reshape(-1, 1)  # 使用 reshape 将其变形为二维数组

        # visit_counts_processed = np.array([arr[:196] if len(arr) > 196 else arr for arr in visit_counts])
        epsilon = 1e-10  # 很小的数值，用于避免除以零
        action_probs = visit_counts_processed / (np.sum(visit_counts_processed, axis=1, keepdims=True) + epsilon)

        # 将value_targets数组中的元素类型转换为浮点数
        value_targets = np.array(value_targets).astype(np.float32)
        model_input = np.array([arr.astype(np.float32) for arr in model_input])

        model_input_tensor = tf.convert_to_tensor(model_input)

        value_targets_tensor = tf.convert_to_tensor(value_targets)
        action_probs_trimmed = action_probs[:value_targets_tensor.shape[0]]
        y_true = tf.concat([value_targets_tensor, action_probs_trimmed], axis=1)

        def custom_loss(y_true, y_pred):
            y_pred_policy = y_pred[0]  # 策略输出
            y_pred_value = y_pred[1]  # 价值输出

            y_true_policy = y_true[:, :-1]  # 所有除了最后一个元素之外
            y_true_value = y_true[:, -1:]  # 只有最后一个元素

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

        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss=custom_loss)
        history = self.model.fit(model_input_tensor, y_true, batch_size=batch_size, epochs=1)

        average_reward = np.mean(value_targets)

        with open("training_log.txt", "a") as file:
            file.write(f"Loss: {history.history['loss'][-1]}, Reward: {average_reward}\n")
# end::zero_train[]
