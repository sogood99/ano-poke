import asyncio
from graphlib import CycleError
from tabulate import tabulate
from poke_env.player.player import *
from poke_env.player.battle_order import *
from poke_env.player.utils import cross_evaluate
from poke_env.player.random_player import RandomPlayer
from poke_env.environment.abstract_battle import *
from poke_env.environment.move import *
from poke_env.environment.pokemon import *
from random import choice


class MaxDamagePlayer(Player):
    def choose_move(self, battle: Battle) -> BattleOrder:
        print("-----------------------------------")
        print([l.status for l in list(battle.team.values())])
        print(list(battle.opponent_team.values()))
        current_mon: Pokemon = battle.active_pokemon
        print((current_mon.stats, current_mon.type_1,
              current_mon.type_2, current_mon.moves, current_mon.boosts, current_mon.active) if current_mon else None)
        mon_moves: Dict[Move] = current_mon.moves if current_mon else None
        first_move: Move = choice(
            list(mon_moves.values())) if mon_moves else None
        if first_move:
            print(first_move.base_power, first_move.type, first_move.boosts,
                  first_move.self_boost, first_move.status, first_move.secondary)
        if battle.available_moves:
            best_move = max(battle.available_moves,
                            key=lambda move: move.base_power)
            return self.create_order(best_move, dynamax=True if battle.can_dynamax else False)
        else:
            return self.choose_random_move(battle)


class MaxDamageModifiedPlayer(Player):
    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        pass


async def main():
    players = [RandomPlayer(max_concurrent_battles=10)] + \
        [MaxDamagePlayer(max_concurrent_battles=10)]

    cross_evaluation = await cross_evaluate(players, n_challenges=100)

    table = [["-"] + [p.username for p in players]]

    print(cross_evaluation)

    for p_1, results in cross_evaluation.items():
        table.append([p_1] + [cross_evaluation[p_1][p_2] for p_2 in results])

    print(tabulate(table))


async def test_human():
    player = MaxDamagePlayer(battle_format="gen8randombattle")
    await player.send_challenges("murkrowa", n_challenges=1)


if __name__ == "__main__":
    # asyncio.get_event_loop().run_until_complete(main())
    asyncio.get_event_loop().run_until_complete(test_human())
