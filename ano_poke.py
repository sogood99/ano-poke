# -*- coding: utf-8 -*-
import numpy as np
import random
import argparse
from tabulate import tabulate

from poke_env.player.utils import *
from poke_env.player.env_player import *
from poke_env.player.random_player import RandomPlayer
from poke_env.environment.move import *
from poke_env.environment.pokemon import *

from torch import nn
from stable_baselines3 import PPO
import stable_baselines3
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy

from gym import spaces

BATTLE_FORMAT = "gen8randombattle"


def transform_action(action: int) -> int:
    if action == -1:
        print("Finished...")
        return -1
    process_action = action - 1
    if process_action >= 4:
        process_action += 12 - 4
    assert \
        -1 <= process_action < 4 or 12 <= process_action < 22, f"Simple assert {process_action}"
    return process_action


def construct_move_vec(move: Move, battle: Battle) -> np.array:
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

        return self._action_to_move(transform_action(action), battle)

    @property
    def observation_space(self):
        return spaces.Box(low=-500, high=500, shape=(10,))

    @property
    def action_space(self):
        return spaces.Discrete(15)


class RlPlayer(SimpleRLEnvPlayer):
    def set_model(self, model: PPO):
        self.model = model

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if self.model is None:
            assert False, "Please set the model before using SimpleRlPlayer"
        obs = self.embed_battle(battle)
        action, _states = self.model.predict(observation=obs)

        return self._action_to_move(transform_action(action), battle)

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        pass


class SimpleRLEnvPlayer2(SimpleRLEnvPlayer):
    def embed_battle(self, battle: Battle):

        # --------------- moves -------------------
        move_vecs = [construct_move_vec(move, battle)
                     for move in battle.available_moves]
        move_vecs += [construct_move_vec(None, None)
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


class RlPlayer2(RlPlayer, SimpleRLEnvPlayer2):
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


class RlPlayer3(RlPlayer, SimpleRLEnvPlayer3):
    pass


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


def ppo_train(env: Player, model: PPO, nb_steps, save_dir):
    print("Training...")
    model.learn(total_timesteps=nb_steps)
    print("Saving...")
    model.save(save_dir)
    print("Saving Finished.")


def ppo_eval(env: Player, model: PPO, nb_episodes):
    env.reset_battles()
    mean_reward, std_reward = evaluate_policy(
        model, model.get_env(), n_eval_episodes=nb_episodes+1)
    print(f"Mean Reward {mean_reward} with std {std_reward}")
    print(f"Player won {env.n_won_battles} out of {env.n_finished_battles-1}")


NB_TRAINING_STEPS = int(5e5)
NB_EVALUATION_EPISODES = 100


if __name__ == "__main__":
    # server_config = ServerConfiguration(
    #     "82.157.5.28:8000", "https://play.pokemonshowdown.com/action.php")
    server_config = None

    # --------------------- argument parser -----------------------

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', type=int,
                        action='store', required=False, help="which argument agent train")
    parser.add_argument('-o', '--opp', type=int,
                        action='store', required=False, help="which opponent", nargs='?')
    parser.add_argument('-l', '--load', action='store_true',
                        required=False)
    parser.add_argument('-c', '--cross_eval',
                        action='store', type=int, nargs='?', const=100, required=False)
    parser.add_argument('-p', '--play',
                        action='store', type=str, required=False, help="opponent to play against")
    args = parser.parse_args()

    # --------------------- main stuff ---------------------------

    ref_list = [SimpleRLEnvPlayer, SimpleRLEnvPlayer2,
                SimpleRLEnvPlayer3]
    if args.train != None:
        assert 0 <= args.train < len(
            ref_list), f"-t must be between {0} and {len(ref_list)}"
        env_player = ref_list[args.train](
            battle_format=BATTLE_FORMAT, server_configuration=server_config)

        # ppo model
        policy_kwargs = dict(activation_fn=nn.ReLU,
                             net_arch=[256, 256, dict(pi=[256, 256], vf=[256, 256])])

        save_dir = "./logs/simple_rl_" + str(args.train)

        if args.load == False:
            print("Initializing new model...")
            model = PPO(MlpPolicy, env=env_player, verbose=1,
                        tensorboard_log=save_dir)
        else:
            print("Loading old model...")
            model = PPO.load(save_dir + ".zip", env=env_player)

    first_opponent = RandomPlayer(battle_format=BATTLE_FORMAT,
                                  server_configuration=server_config)

    second_opponent = MaxDamagePlayer(
        battle_format=BATTLE_FORMAT, server_configuration=server_config)

    third_opponent_model = PPO.load("./logs/simple_rl_0.zip")
    third_opponent = RlPlayer(
        battle_format=BATTLE_FORMAT, server_configuration=server_config)
    third_opponent.set_model(third_opponent_model)

    fourth_opponent_model = PPO.load("./logs/simple_rl_1.zip")
    fourth_opponent = RlPlayer2(
        battle_format=BATTLE_FORMAT, server_configuration=server_config)
    fourth_opponent.set_model(fourth_opponent_model)

    fifth_opponent_model = PPO.load("./logs/simple_rl_2.zip")
    fifth_opponent = RlPlayer3(
        battle_format=BATTLE_FORMAT, server_configuration=server_config)
    fifth_opponent.set_model(fifth_opponent_model)

    opponent_list = [first_opponent, second_opponent,
                     third_opponent, fourth_opponent, fifth_opponent]
    if args.train != None:
        if args.opp != None:
            assert 0 <= args.opp < len(
                opponent_list), f"Opponent must be betweent {0} and {len(opponent_list)}"
            opp_idx = args.opp
        else:
            opp_idx = 1
        opp = opponent_list[opp_idx]
        env_player.play_against(
            env_algorithm=ppo_train,
            opponent=opp,
            env_algorithm_kwargs={"model": model,
                                  "nb_steps": NB_TRAINING_STEPS,
                                  "save_dir": save_dir+".zip"},
        )

        print("Results against random player:")
        env_player.play_against(
            env_algorithm=ppo_eval,
            opponent=first_opponent,
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

    if args.cross_eval != None:
        print("Starting cross evaluation...")

        async def cross_eval_opps():
            cross_evaluation = await cross_evaluate(opponent_list, n_challenges=args.cross_eval)
            table = [["-"] + [p.username for p in opponent_list]]

            for p_1, results in cross_evaluation.items():
                table.append([p_1] + [cross_evaluation[p_1][p_2]
                                      for p_2 in results])

            print(tabulate(table))

        asyncio.get_event_loop().run_until_complete(cross_eval_opps())

    if args.play != None:
        print(f"Sending challenge to {args.play}...")

        async def play_human():
            if args.train == None:
                opp_idx = -1
            else:
                opp_idx = args.train

            await opponent_list[opp_idx].send_challenges(args.p, n_challenges=1)

        asyncio.get_event_loop().run_until_complete(play_human())
