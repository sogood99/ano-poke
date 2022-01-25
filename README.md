# ano-poke

Pokémon Showdown RL agent. Plans:

- PPO training for Deep Recurrent RL.
- Observation consists of party current pokemon's stats (total health , type info, health percentage, move infos, buffs, debuffs), opponent current pokemon stats (same as current pokemon), opponent last move information, party pokemon's brief information (type, total health, health percentage), and if dynamaxable.
- Actions consist of the a probability distribution of moves, dynamax, and pokemon. Use softmax output.

- Observation thoughts:

  - For each pokemon, 18 dim for pokemon type (one hot), 6 for total stats, 2 for HP and current hp percent, 2 for status, 1 for if active (total 29 per mon)
  - For enemy pokemon, 18 dim for pokemon type, 2 for health and health percentage, 2 for status, 1 for if active (total 23)
  - For current pokemon, 1 more for if it is dynamaxed (total 1)
  - $7*2$ more slots for self and enemy current pokemon boosts (total 14)
  - For each available moves, needs 18 for type, 1 for base attack, 1 for attack multiplier, 1 for accuracy, 3 for move category (physical, special, status), 1 for if inflicts status, 1 for percentage heal, 5 for self boost 5 for enemy boost. Each available move has possible secondary effects, take the one with highest probability and 1 slot for status, 7 for self boost and 7 for enemy boost (total 54 per move)
  - All in all, totals to $6 * 29+6 * 23+1+14+4 * 54=543$ input dimensions.

- Reward thoughts:

  - +5 for fainting enemy pokemon.
  - -5 for fainting in self party pokemon.
  - +10 for victory.
  - -10 for loss.
  - +0.1 for % stat buff / enemy stat debuff
  - -0.1 for % stat debuff / enemy stat buff
  - +% health of enemy pokemon dealt
  - -% heath of party pokemon dealt

  To let the agent learn the basics of pokemon for the first few iterations

  - -20 for choosing a move and having > $\varepsilon$ probability for switching or visa versa
  - -20 for having > $\varepsilon$ probability for fainted pokemon.
  - -20 for dynamaxing but probability distribution for switching > $\varepsilon$.

- Output thoughts:
  - 4 for moves w/o dynamax, 4 for moves with dynamax, 6 dimensions for switching pokemon $= 14$ total.
