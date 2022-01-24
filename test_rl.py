# -*- coding: utf-8 -*-
from matplotlib.pyplot import step
import numpy as np

from poke_env.player.env_player import *
from poke_env.player.random_player import RandomPlayer

from stable_baselines3 import PPO
import stable_baselines3
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

from gym import spaces


class SimpleRLPlayer(Gen8EnvSinglePlayer):
    def embed_battle(self, battle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )

        remaining_mon_team = (
            len([mon for mon in battle.team.values() if mon.fainted]) / 6
        )
        remaining_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        return np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [remaining_mon_team, remaining_mon_opponent],
            ], dtype=float
        )

    def compute_reward(self, battle) -> float:
        return self.reward_computing_helper(battle, fainted_value=2, hp_value=1, status_value=0.2, victory_value=30)

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=float("inf"), shape=(10,))

    @property
    def action_space(self):
        return spaces.Discrete(len(super().action_space))


class MaxDamagePlayer(RandomPlayer):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves,
                            key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)


def ppo_train(env, model: PPO, nb_steps):
    model.learn(total_timesteps=nb_steps)
    model.save("./logs/test_ppo_model")


def ppo_eval(env: Player, model: PPO, nb_episodes):
    env.reset_battles()
    mean_reward, std_reward = evaluate_policy(
        model, model.get_env(), n_eval_episodes=nb_episodes)
    print(f"Mean Reward {mean_reward} with std {std_reward}")
    print(f"Player won {env.n_won_battles} out of {env.n_finished_battles}")


NB_TRAINING_STEPS = int(2e5)
NB_EVALUATION_EPISODES = 100


if __name__ == "__main__":
    env_player = SimpleRLPlayer(battle_format="gen8randombattle")

    opponent = RandomPlayer(battle_format="gen8randombattle")
    second_opponent = MaxDamagePlayer(battle_format="gen8randombattle")

    # ppo model

    model = PPO(MlpPolicy, env=env_player, verbose=1,
                tensorboard_log="./logs/test_ppo")
    print("Training")
    env_player.play_against(
        env_algorithm=ppo_train,
        opponent=second_opponent,
        env_algorithm_kwargs={"model": model,
                              "nb_steps": NB_TRAINING_STEPS},
    )

    print("Results against random player:")
    env_player.play_against(
        env_algorithm=ppo_eval,
        opponent=opponent,
        env_algorithm_kwargs={"model": model,
                              "nb_episodes": NB_EVALUATION_EPISODES},
    )

    print("\nResults against max player:")
    env_player.play_against(
        env_algorithm=ppo_eval,
        opponent=second_opponent,
        env_algorithm_kwargs={"model": model,
                              "nb_episodes": NB_EVALUATION_EPISODES},
    )
