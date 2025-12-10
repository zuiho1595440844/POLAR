# tag::dl_agent_imports[]
import numpy as np

from dlgo.agent.base import Agent
from dlgo.agent.helpers import is_point_an_eye
#from dlgo import encoders
from dlgo import goboard
from dlgo import kerasutil
# end::dl_agent_imports[]
__all__ = [
    'DeepLearningAgent',
    'load_prediction_agent',
]


# tag::dl_agent_init[]
from dlgo.goboard import Move


class DeepLearningAgent(Agent):
    def __init__(self, model, encoder):
        Agent.__init__(self)
        self.model = model
        self.encoder = encoder
# end::dl_agent_init[]

# tag::dl_agent_predict[]
    def predict(self, game_state):
        encoded_state = self.encoder.encode(game_state)
        input_tensor = np.array([encoded_state])
        policy_output, value_output = self.model.predict(input_tensor)
        return policy_output[0], value_output[0]

    def select_move(self, game_state):
        num_moves = self.encoder.num_moves()  # 使用 ZeroEncoder 的 num_moves
        move_probs, _ = self.predict(game_state)  # 假设 predict 返回策略和价值
        # print(move_probs)
        # 确保 move_probs 是一维的
        move_probs = move_probs.flatten()

        # 根据 ZeroEncoder 对 move_probs 进行处理
        move_probs = move_probs ** 3
        eps = 1e-6
        move_probs = np.clip(move_probs, eps, 1 - eps)
        move_probs = move_probs / np.sum(move_probs)

        # 根据概率选择移动
        candidates = np.arange(num_moves)
        ranked_moves = np.random.choice(candidates, num_moves, replace=False, p=move_probs)
        for point_idx in ranked_moves:
            move = self.encoder.decode_move_index(point_idx)  # 使用 ZeroEncoder 的 decode_move_index
            if game_state.is_valid_move(move):
                return move
        return Move.pass_turn()

    # <1> Turn the probabilities into a ranked list of moves.
# <2> Sample potential candidates
# <3> Starting from the top, find a valid move that doesn't reduce eye-space.
# <4> If no legal and non-self-destructive moves are left, pass.
# end::dl_agent_candidates[]

# tag::dl_agent_serialize[]
    def serialize(self, h5file):
        h5file.create_group('encoder')
        h5file['encoder'].attrs['name'] = self.encoder.name()
        h5file['encoder'].attrs['board_width'] = self.encoder.board_width
        h5file['encoder'].attrs['board_height'] = self.encoder.board_height
        h5file.create_group('model')
        kerasutil.save_model_to_hdf5_group(self.model, h5file['model'])
# end::dl_agent_serialize[]


# tag::dl_agent_deserialize[]
def load_prediction_agent(h5file):
    model = kerasutil.load_model_from_hdf5_group(h5file['model'])
    encoder_name = h5file['encoder'].attrs['name']
    if not isinstance(encoder_name, str):
        encoder_name = encoder_name.decode('ascii')
    board_width = h5file['encoder'].attrs['board_width']
    board_height = h5file['encoder'].attrs['board_height']
    encoder = encoders.get_encoder_by_name(
        encoder_name, (board_width, board_height))
    return DeepLearningAgent(model, encoder)
# tag::dl_agent_deserialize[]
