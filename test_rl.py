# -*- coding: utf-8 -*-
import numpy as np

from poke_env.player.env_player import *
from poke_env.player.random_player import RandomPlayer
from poke_env.environment.move import *
from poke_env.environment.pokemon import *

from torch import nn
from stable_baselines3 import PPO
import stable_baselines3
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

from gym import spaces


class SimpleRLEnvPlayer(Gen8EnvSinglePlayer):
    def embed_battle(self, battle: AbstractBattle):
        moves_base_power = np.zeros(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = move.base_power / 100
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )

        remaining_mon_team = len(
            [mon for mon in battle.team.values() if mon.fainted]) / 6
        remaining_mon_opponent = len(
            [mon for mon in battle.opponent_team.values() if mon.fainted]) / 6

        return np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [remaining_mon_team, remaining_mon_opponent],
            ], dtype=float
        )

    def compute_reward(self, battle) -> float:
        return self.reward_computing_helper(battle, fainted_value=2, hp_value=1, status_value=0.2, victory_value=30)

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if battle not in self._observations or battle not in self._actions:
            self._init_battle(battle)
        self._observations[battle].put(self.embed_battle(battle))
        action = self._actions[battle].get()
        if action > 4:
            action += 12

        return self._action_to_move(action, battle)

    @property
    def observation_space(self):
        return spaces.Box(low=-500, high=500, shape=(10,))

    @property
    def action_space(self):
        return spaces.Discrete(14)


class RlPlayer(SimpleRLEnvPlayer):
    def set_model(self, model: PPO):
        self.model = model

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if self.model is None:
            assert False, "Please set the model before using SimpleRlPlayer"
        obs = self.embed_battle(battle)
        action, _states = self.model.predict(observation=obs)

        return self._action_to_move(action, battle)

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        pass


class SimpleRLEnvPlayer2(SimpleRLEnvPlayer):
    def embed_battle(self, battle: Battle):

        # --------------- moves -------------------
        def construct_move_vec(move: Move) -> np.array:
            if move == None:
                return np.zeros(40)
            move_base_power = move.base_power / 100
            move_accuracy = move.accuracy

            move_type_vec = np.zeros(18)
            move_dmg_mult = move.expected_hits
            if move.type:
                move_dmg_mult *= move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )
                move_type_vec[move.type.value - 1] = 1

            move_category = np.zeros(3)
            move_category[move.category.value - 1] = 1

            move_status = 1. if move.status != None or move.volatile_status != None else 0.

            move_heal = move.heal

            def get_boost(boost_dict: Dict) -> np.array:
                boost_arr = np.zeros(7)
                if (boost_dict == None):
                    return boost_arr
                boost_arr[0] = boost_dict.get("spd", 0)
                boost_arr[1] = boost_dict.get("atk", 0)
                boost_arr[2] = boost_dict.get("def", 0)
                boost_arr[3] = boost_dict.get("spa", 0)
                boost_arr[4] = boost_dict.get("spd", 0)
                boost_arr[5] = boost_dict.get("accuracy", 0)
                boost_arr[6] = boost_dict.get("evasion", 0)
                return boost_arr

            move_self_boost = get_boost(move.self_boost)
            move_enemy_boost = get_boost(move.boosts)

            return np.concatenate([move_type_vec, [move_base_power, move_dmg_mult, move_accuracy], move_category, [move_status, move_heal], move_self_boost, move_enemy_boost])

        move_vecs = [construct_move_vec(move)
                     for move in battle.available_moves]
        move_vecs += [construct_move_vec(None)
                      for _ in range(4-len(battle.available_moves))]

        # party pokemon
        active_mon_type = np.zeros(18)
        active_mon_type[battle.active_pokemon.type_1.value-1] = 1
        if battle.active_pokemon.type_2 is not None:
            active_mon_type[battle.active_pokemon.type_2.value-1] = 1

        enemy_active_mon_type = np.zeros(18)
        enemy_active_mon_type[battle.opponent_active_pokemon.type_1.value-1] = 1
        if battle.opponent_active_pokemon.type_2 is not None:
            active_mon_type[battle.opponent_active_pokemon.type_2.value-1] = 1

        remaining_mon_team = len(
            [mon for mon in battle.team.values() if mon.fainted]) / 6
        remaining_mon_opponent = len(
            [mon for mon in battle.opponent_team.values() if mon.fainted]) / 6

        return np.concatenate(
            [
                *move_vecs,
                active_mon_type,
                enemy_active_mon_type,
                [remaining_mon_team, remaining_mon_opponent],
            ], dtype=float
        )

    @property
    def observation_space(self):
        return spaces.Box(low=-500, high=500, shape=(198,))


class RlPlayer2(SimpleRLEnvPlayer2):
    def set_model(self, model: PPO):
        self.model = model

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if self.model is None:
            assert False, "Please set the model before using SimpleRlPlayer"
        obs = self.embed_battle(battle)
        action, _states = self.model.predict(observation=obs)

        return self._action_to_move(action, battle)

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        pass


class SimpleRLEnvPlayer3(SimpleRLEnvPlayer):

    def embed_battle(self, battle: Battle):

        # ------------------ pokemon stuff ---------------------
        def get_mon_vec(mon: Pokemon, is_enemy=False):
            if mon == None:
                return np.zeros(30, dtype=np.float64) if is_enemy == False else np.zeros(21, dtype=np.float64)

            mon_type_vec = np.zeros(18, dtype=np.float64)
            if mon.types != None:
                mon_type_vec[mon.type_1.value-1] = 1
                if mon.type_2 != None:
                    mon_type_vec[mon.type_2.value-1] = 1

            if is_enemy == False:
                mon_stat_vec = np.zeros(7)
                mon_stat_vec[0] = mon.stats.get("accuracy", 0) / 100
                mon_stat_vec[1] = mon.stats.get("atk", 0) / 100
                mon_stat_vec[2] = mon.stats.get("def", 0) / 100
                mon_stat_vec[3] = mon.stats.get("evasion", 0) / 100
                mon_stat_vec[4] = mon.stats.get("spa", 0) / 100
                mon_stat_vec[5] = mon.stats.get("spd", 0) / 100
                mon_stat_vec[6] = mon.stats.get("spe", 0) / 100
                mon_total_hp = mon.max_hp / 100
                mon_current_hp = mon.current_hp / 100

            mon_has_status = 1.0 if isinstance(
                mon.status, Status) and mon.status != Status.FNT else 0.0
            mon_is_faint = 1.0 if mon.status is Status.FNT else 0.0

            mon_is_active = 1.0 if mon.active == True else 0.0

            if is_enemy == False:
                return np.concatenate([mon_type_vec, mon_stat_vec, [mon_total_hp, mon_current_hp, mon_has_status, mon_is_faint, mon_is_active]], dtype=np.float64)
            else:
                return np.concatenate([mon_type_vec, [mon_has_status, mon_is_faint, mon_is_active]], dtype=np.float64)

        mon_vecs = [get_mon_vec(mon) for mon in battle.team.values()]
        mon_vecs += [get_mon_vec(mon, is_enemy=True)
                     for mon in battle.opponent_team.values()]
        mon_vecs += [get_mon_vec(None, is_enemy=True)
                     for _ in range(6-len(battle.opponent_team))]

        current_mon_is_dyna = 1.0 if battle.active_pokemon.is_dynamaxed else 0.0

        # ------------------ moves stuff ---------------------

        moves_base_power = np.zeros(4, dtype=np.float64)
        moves_dmg_multiplier = np.ones(4, dtype=np.float64)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = move.base_power / 100
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )

        return np.concatenate(
            [
                *mon_vecs,
                moves_base_power,
                moves_dmg_multiplier,
                [current_mon_is_dyna]
            ], dtype=np.float64)

    @property
    def observation_space(self):
        return spaces.Box(low=-500, high=500, shape=(315,))


class MaxDamagePlayer(Player):
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
    model.save("./logs/simple_rl_3")


def ppo_eval(env: Player, model: PPO, nb_episodes):
    env.reset_battles()
    mean_reward, std_reward = evaluate_policy(
        model, model.get_env(), n_eval_episodes=nb_episodes)
    print(f"Mean Reward {mean_reward} with std {std_reward}")
    print(f"Player won {env.n_won_battles} out of {env.n_finished_battles}")


def ppo_play(player: EnvPlayer, model: PPO):
    env = model.get_env()
    obs = env.reset()
    while True:
        action, _states = model.predict(observation=obs, deterministic=True)
        obs, reward, done, info = env.step(action)


NB_TRAINING_STEPS = int(2e5)
NB_EVALUATION_EPISODES = 100


if __name__ == "__main__":
    # server_config = ServerConfiguration(
    #     "82.157.5.28:8000", "https://play.pokemonshowdown.com/action.php")
    server_config = None

    env_player = SimpleRLEnvPlayer3(
        battle_format="gen8randombattle", server_configuration=server_config)

    opponent = RandomPlayer(battle_format="gen8randombattle",
                            server_configuration=server_config)
    second_opponent = MaxDamagePlayer(
        battle_format="gen8randombattle", server_configuration=server_config)

    third_opponent_model = opp_model = PPO.load("./logs/simple_rl_1.zip")
    third_opponent = RlPlayer(
        battle_format="gen8randombattle", server_configuration=server_config)
    third_opponent.set_model(third_opponent_model)

    fourth_opponent_model = opp_model = PPO.load("./logs/simple_rl_2.zip")
    fourth_opponent = RlPlayer2(
        battle_format="gen8randombattle", server_configuration=server_config)
    fourth_opponent.set_model(fourth_opponent_model)

    # ppo model
    policy_kwargs = dict(activation_fn=nn.ReLU,
                         net_arch=[256, 256, dict(pi=[256, 256], vf=[256, 256])])
    # model = PPO(MlpPolicy, env=env_player, verbose=1,
    # tensorboard_log="./logs/simple_rl_3")
    model = PPO.load("./logs/simple_rl_3.zip", env=env_player)

    print(f"Training")
    env_player.play_against(
        env_algorithm=ppo_train,
        opponent=third_opponent,
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

    print("\nResults against basic type damage player:")
    env_player.play_against(
        env_algorithm=ppo_eval,
        opponent=third_opponent,
        env_algorithm_kwargs={"model": model,
                              "nb_episodes": NB_EVALUATION_EPISODES},
    )

    print("\nResults against intermediate type damage player:")
    env_player.play_against(
        env_algorithm=ppo_eval,
        opponent=fourth_opponent,
        env_algorithm_kwargs={"model": model,
                              "nb_episodes": NB_EVALUATION_EPISODES},
    )

    # async def test_human():
    #     await trained_agent.send_challenges("murkrowa", n_challenges=1)

    # asyncio.get_event_loop().run_until_complete(test_human())
