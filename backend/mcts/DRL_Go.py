# 仿照zero
from keras.layers import Conv2D, Dense, Flatten, Input
from dlgo import scoring
from dlgo import zero
from dlgo.goboard import GameState, Player, Point
import model



# tag::zero_simulate[]
def simulate_game(board_size, black_agent, black_collector, white_agent, white_collector):
    print('Starting the game!')
    game = GameState.new_game(board_size)
    black_agent.initialize_game(game)
    white_agent.initialize_game(game)
    agents = {
        Player.black: black_agent,
        Player.white: white_agent,
    }

    black_collector.begin_episode()
    white_collector.begin_episode()

    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        game = game.apply_move(next_move)

        # 在每一步之后进行Sarsa(λ)更新
        if game.next_player == Player.white:
            black_agent.update_with_sarsa_lambda(black_agent.root, gamma=0.9, lambda_=0.5)
        else:
            white_agent.update_with_sarsa_lambda(white_agent.root, gamma=0.9, lambda_=0.5)

    # 游戏结束，进行Q-Learning更新
    black_agent.update_with_q_learning(black_agent.root, gamma=0.9)
    white_agent.update_with_q_learning(white_agent.root, gamma=0.9)

    game_result = scoring.compute_game_result(game)
    # 输出Q值统计信息
    score_difference = game_result.b - game_result.w
    captures_difference = game.board.num_captures(Player.black) - game.board.num_captures(Player.white)
    # 根据目数差和吃子数差来调整奖励的规模
    final_reward = score_difference + captures_difference
    with open("training_log.txt", "a") as file:
        file.write(f"Game result: {game_result}\n")
    black_collector.complete_episode(final_reward)
    white_collector.complete_episode(-final_reward)
    del game

# end::zero_simulate[]


# tag::zero_model[]
board_size = 19
encoder = zero.ZeroEncoder(board_size)
print(encoder.shape())
board_input = Input(shape=encoder.shape(), name='board_input')
# model = model.modified_vgg16_model(encoder.shape())
# model = model.lightweight_vgg16_model_op(encoder.shape())
# print(model.summary())
model = model.UCM_GoNet_TF(encoder.shape())
model.summary()
# # end::zero_model[]

# tag::zero_train[]
black_agent = zero.ZeroAgent(
    model, encoder, rounds_per_move=100, c=2.0)  # <4>
white_agent = zero.ZeroAgent(
    model, encoder, rounds_per_move=100, c=2.0)
c1 = zero.ZeroExperienceCollector()
c2 = zero.ZeroExperienceCollector()
black_agent.set_collector(c1)
white_agent.set_collector(c2)


for i in range(10000):
    simulate_game(board_size, black_agent, c1, white_agent, c2)
    with open("training_log.txt", "a") as file:
        file.write(f"Game {i + 1} completed\n")
    exp = zero.combine_experience([c1, c2])
    black_agent.train(exp, 0.001, 16)
    if i % 5 == 0:
        black_agent.model.save("mix_model_" + str(i) + ".hdf5")
