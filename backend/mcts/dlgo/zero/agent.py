import numpy as np
from keras.optimizers import Adam
from ..agent import Agent
import tensorflow as tf

__all__ = [
    'ZeroAgent',
]

# tag::branch_struct[]
from ..goboard import Move
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
        self.Q = value
        self.parent = parent
        self.last_move = last_move
        self.total_visit_count = 1
        self.branches = {}
        for move, p in priors.items():
            if move == Move.pass_turn() or state.is_valid_move(move):
                self.branches[move] = Branch(p)
        self.branches[Move.pass_turn()] = Branch(0.0)  # 假设“pass”动作的先验概率为0
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

    # 根据当前状态和动作计算即时奖励 R(s,a)
    def calculate_R_s_a(self, game_state):
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

    def update_with_sarsa_lambda(self, node, gamma, lambda_, alpha=0.1):
        """
        使用Sarsa(λ)进行更新。
        """
        while node is not None and node.parent is not None:
            next_node = node.parent
            next_move = next_node.last_move
            if next_move is not None:
                # 计算 TD 误差 delta
                R_s_a = self.calculate_R_s_a(node.state)  # 确保奖励函数符合任务目标
                next_q = self.get_next_Q(next_node, next_move)
                delta = R_s_a + gamma * next_q - node.Q

                # 更新资格迹
                if next_move not in node.E:
                    node.E[next_move] = 0
                node.E[next_move] = gamma * lambda_ * node.E[next_move] + 1

                # 使用资格迹传播更新 Q 值
                node.Q += alpha * delta * node.E[next_move]
            node = next_node

    def update_with_q_learning(self, node, gamma, alpha=0.1):
        """
        使用Q-Learning进行更新。
        """
        while node is not None:
            move = node.last_move
            if move is not None:
                max_next_q = self.get_max_next_Q(node)
                # 按照Q-Learning的贝尔曼方程更新
                reward = self.calculate_R_s_a(node.state)
                node.Q = node.Q + alpha * (reward + gamma * max_next_q - node.Q)
            node = node.parent

    def get_next_Q(self, node, next_move):
        """
        获取下一个动作的Q值。
        """
        if next_move in node.children:
            child_node = node.children[next_move]
            return child_node.Q
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
        root = self.create_node(game_state)
        for _ in range(self.num_rounds):
            node = root
            next_move = self.select_branch(node)
            while node.has_child(next_move):
                node = node.get_child(next_move)
                next_move = self.select_branch(node)

            # 执行动作并创建新节点
            new_state = node.state.apply_move(next_move)
            child_node = self.create_node(new_state, next_move, node)

            # for parent, move in reversed(path):
            #     # Q-Learning 更新
            #     self.update_with_q_learning(parent, gamma=0.9, alpha=0.1)

            # 使用反向传播更新节点
            value = -child_node.Q
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
                for idx in range(self.encoder.num_moves())
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

    def select_branch(self, node, c_puct=1.0):
        """
        使用UCB公式选择最佳分支。
        """
        total_n = sum(branch.visit_count for branch in node.branches.values())
        best_score = -float('inf')
        best_move = None
        for move, branch in node.branches.items():
            Q = node.Q  # 使用累积长期价值
            P = branch.prior
            N = branch.visit_count
            U = c_puct * P * np.sqrt(total_n) / (1 + N)
            score = Q + U
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def create_node(self, game_state, move=None, parent=None):
        """
        创建新的搜索树节点。
        """
        state_tensor = self.encoder.encode(game_state)
        model_input = np.array([state_tensor])
        # print(model_input.shape)
        priors, values = self.model.predict(model_input)
        value = values[0][0]  # 作为累积长期价值
        priors = np.squeeze(priors)
        move_priors = {self.encoder.decode_move_index(i): p for i, p in enumerate(priors)}
        return ZeroTreeNode(game_state, value, move_priors, parent, move)

    # end::zero_create_node[]

    # tag::zero_train[]
    def train(self, experience, learning_rate, batch_size):
        print(f"Training with rewards: {experience.rewards}")
        with open("training_log.txt", "a") as file:
            file.write(f"Training with rewards: {experience.rewards}\n")
        num_examples = experience.states.shape[0]

        if num_examples == 0:
            print("Experience data is empty. Skipping training.")
            return

        model_input = experience.states
        visit_sums = np.sum(experience.visit_counts, axis=1).reshape((-1, 1))
        # with open("training_log.txt", "a") as file:
        #     file.write(f"visit_counts:{experience.visit_counts},Visit_sums:{visit_sums}\n")
        visit_sums = np.where(visit_sums == 0, 1e-10, visit_sums)  # 避免除以零

        action_probs = experience.visit_counts / visit_sums  # π_i，动作概率
        value_targets = experience.rewards.reshape((-1, 1))  # z_i，值目标

        # 在模型训练之前计算平均奖励值
        average_reward = np.mean(value_targets)

        # 在训练之后，记录包含平均奖励值的日志
        print(f"Average reward: {average_reward}")
        with open("training_log.txt", "a") as file:
            file.write(f"Average reward: {average_reward}\n")

        # 定义自定义损失函数
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

        # 合并值目标和动作概率为y_true
        y_true = tf.concat([value_targets, action_probs], axis=1)

        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss=custom_loss)
        history = self.model.fit(model_input, y_true, batch_size=batch_size, epochs=10)

        with open("training_log.txt", "a") as file:
            file.write(f"Loss: {history.history['loss'][-1]}, Reward: {np.mean(value_targets)}\n")

    def train(self, experience, learning_rate, batch_size):
        print(f"Training with rewards: {experience.rewards}")
        with open("training_log.txt", "a") as file:
            file.write(f"Training with rewards: {experience.rewards}\n")
        num_examples = experience.states.shape[0]

        if num_examples == 0:
            print("Experience data is empty. Skipping training.")
            return

        model_input = experience.states
        visit_sums = np.sum(experience.visit_counts, axis=1).reshape((-1, 1))
        visit_sums = np.where(visit_sums == 0, 1e-10, visit_sums)  # 避免除以零

        action_probs = experience.visit_counts / visit_sums  # π_i，动作概率
        value_targets = experience.rewards.reshape((-1, 1))  # z_i，值目标

        # 合并值目标和动作概率为y_true
        y_true_policy = action_probs
        y_true_value = value_targets

        # 编译模型，使用两个不同的损失函数
        self.model.compile(optimizer=Adam(learning_rate=learning_rate),
                           loss=['categorical_crossentropy', 'mean_squared_error'])

        # 训练模型，假设模型有两个输出：策略输出和价值输出
        history = self.model.fit(model_input, [y_true_policy, y_true_value],
                                 batch_size=batch_size, epochs=10)

        with open("training_log.txt", "a") as file:
            file.write(f"Loss: {history.history['loss'][-1]}, Reward: {np.mean(value_targets)}\n")

# end::zero_train[]
