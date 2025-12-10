import random
import numpy as np
from keras.optimizers import Adam
from tensorflow.python.keras.losses import MeanSquaredError
from ..agent import Agent
import tensorflow as tf


__all__ = [
    'ZeroAgent',
]

from ..goboard import Move
from ..gotypes import Player, Point
from ..utils import print_move


class Branch:
    def __init__(self, prior):
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0


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
        self.E = {move: 0.0 for move in self.branches}

    def moves(self):
        return self.branches.keys()

    def add_child(self, move, child_node):
        self.children[move] = child_node

    def has_child(self, move):
        return move in self.children

    def get_child(self, move):
        return self.children[move]

    def record_visit(self, move, value):
        if move not in self.branches:
            self.branches[move] = Branch(0.0)
        self.total_visit_count += 1
        self.branches[move].visit_count += 1
        self.branches[move].total_value += value

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


class ZeroAgent(Agent):
    def __init__(self, model, encoder, rounds_per_move=1600, c=2.0):
        self.model = model
        self.encoder = encoder
        self.collector = None
        self.num_rounds = rounds_per_move
        self.c = c

    def initialize_game(self, initial_state):
        self.root = self.create_node(initial_state)

    def calculate_R_s_a(self, game_state, move):
        """
        计算在给定状态下执行某个动作的奖励值。
        """

        def count_continuous_stones(board, player, point, direction):
            """
            计算在指定方向上连续的棋子数量。
            """
            count = 0
            row, col = point.row, point.col
            dr, dc = direction

            while board.get(Point(row, col)) == player:
                count += 1
                row += dr
                col += dc

            return count

        def evaluate_point(board, player, point):
            """
            评估一个点的得分，根据它在不同方向上连续棋子的数量。
            """
            directions = [
                (1, 0),  # 水平方向
                (0, 1),  # 垂直方向
                (1, 1),  # 右下方向
                (1, -1)  # 左下方向
            ]

            score = 0

            for dr, dc in directions:
                continuous_count = count_continuous_stones(board, player, point, (dr, dc))
                continuous_count += count_continuous_stones(board, player, point, (-dr, -dc)) - 1

                if continuous_count >= 5:
                    score += 100  # 成五，加大量分数
                elif continuous_count == 4:
                    score += 50  # 成四，加高分
                elif continuous_count == 3:
                    score += 10  # 成三，加分
                elif continuous_count == 2:
                    score += 1  # 成二，加少量分数

            return score

        board = game_state.board
        player = game_state.next_player
        point = move.point

        if not board.is_on_grid(point) or not board.is_empty(point):
            return 0  # 非法动作

        return evaluate_point(board, player, point)

    def update_with_sarsa_lambda(self, node, gamma, lambda_):
        while node is not None and node.parent is not None:
            next_node = node.parent
            next_move = next_node.last_move
            if next_move is not None:
                R_s_a = self.calculate_R_s_a(node.state, next_move)
                next_q = self.get_next_Q(next_node, next_move)
                node.E[next_move] = R_s_a + gamma * next_q - node.value
                node.value += lambda_ * node.E[next_move]
            node = next_node

    def update_with_q_learning(self, node, gamma):
        while node is not None:
            move = node.last_move
            if move is not None:
                max_next_q = self.get_max_next_Q(node)
                node.value += gamma * (max_next_q - node.value)
            node = node.parent

    def get_next_Q(self, node, next_move):
        if next_move in node.children:
            child_node = node.children[next_move]
            return child_node.value
        return 0

    def get_max_next_Q(self, node):
        if not node.children:
            return 0
        return max(child.value for child in node.children.values())

    def select_move(self, game_state):
        root = self.create_node(game_state)
        for _ in range(self.num_rounds):
            node = root
            legal_moves = game_state.legal_moves()
            if len(legal_moves) == 0:
                return Move.resign()
            next_move = self.select_branch(node, legal_moves)
            while node.has_child(next_move):
                node = node.get_child(next_move)
                next_move = self.select_branch(node)
            new_state = node.state.apply_move(next_move)
            child_node = self.create_node(new_state, next_move, node)
            value = -child_node.value
            while node is not None:
                node.record_visit(next_move, value)
                value = -value
                node = node.parent

        if self.collector is not None:
            root_state_tensor = self.encoder.encode(game_state)
            visit_counts = np.array([
                root.visit_count(
                    self.encoder.decode_move_index(idx))
                for idx in range(len(legal_moves))
            ])
            self.collector.record_decision(
                root_state_tensor, visit_counts)

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

    def select_branch(self, node, legal_moves, c_puct=1.25):
        best_move = None
        best_score = -float('inf')
        total_n = sum(branch.visit_count for branch in node.branches.values())
        for move, branch in node.branches.items():
            Q = node.value
            P = branch.prior
            N = branch.visit_count
            U = c_puct * P * np.sqrt(total_n) / (1 + N)
            score = Q + U
            if score > best_score:
                best_score = score
                best_move = move
        if best_move is None and legal_moves:
            best_move = random.choice(legal_moves)
        return best_move

    def create_node(self, game_state, move=None, parent=None):
        state_tensor = self.encoder.encode(game_state)
        model_input = np.array([state_tensor])
        priors, values = self.model.predict(model_input)
        value = values[0][0]
        priors = np.squeeze(priors)
        legal_moves = game_state.legal_moves()
        move_priors = {move: priors[i] for i, move in enumerate(legal_moves)}
        return ZeroTreeNode(game_state, value, move_priors, parent, move)

    def train(self, experience, learning_rate, batch_size):
        print(f"Training with rewards: {experience.rewards}")
        num_examples = experience.states.shape[0]

        if num_examples == 0:
            print("Experience data is empty. Skipping training.")
            return

        model_input = experience.states
        value_targets = experience.rewards.reshape((-1, 1))
        visit_counts = np.array(experience.visit_counts)
        visit_counts_processed = np.array([vc[0] for vc in visit_counts])  # 展开嵌套数组
        epsilon = 1e-10
        action_probs = visit_counts_processed / (np.sum(visit_counts_processed, keepdims=True) + epsilon)
        value_targets = np.array(value_targets).astype(np.float32)
        model_input = np.array([arr.astype(np.float32) for arr in model_input])

        model_input_tensor = tf.convert_to_tensor(model_input)
        value_targets_tensor = tf.convert_to_tensor(value_targets)
        action_probs_tensor = tf.convert_to_tensor(action_probs, dtype=tf.float32)

        # 使用内置的损失函数
        self.model.compile(optimizer=Adam(learning_rate=learning_rate),
                           loss=[tf.keras.losses.MeanSquaredError(), tf.keras.losses.CategoricalCrossentropy()])




        history = self.model.fit(model_input_tensor, [value_targets_tensor, action_probs_tensor], batch_size=batch_size,
                                 epochs=1)

        average_reward = np.mean(value_targets)

        with open("training_log.txt", "a") as file:
            file.write(f"Loss: {history.history['loss'][-1]}, Reward: {average_reward}\n")
