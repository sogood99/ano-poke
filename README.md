# ano-poke

Pokémon Showdown RL agent. Plans:

- PPO training for Deep Recurrent RL.
- Observation consists of party current pokemon's stats (total health , type info, health percentage, move infos, buffs, debuffs), opponent current pokemon stats (mirror of current pokemon), opponent last move information, party pokemon's breif information (type, total health, health percentage), and if dynamaxable.
- Actions consist of the a probability distribution of moves, dynamax, and pokemon. Use softmax output.

- Observation thoughts:

  - For each pokemon, 18 dim for pokemon type (one hot), 6 for base stats, 2 for HP and current hp percent (total 26 per mon)
  - For enemy pokemon, 18 dim for pokemon type (total 18)
  - $7*2$ more slots for self and enemy current pokemon boosts (total 14)
  - For each available moves, needs 18 for type, 1 for base attack, 1 for is inflicts status, 7 for self boost 7 for enemy boost. Each available move has possible secondary effects, take the one with highest probability and 1 slot for status, 7 for self boost and 7 for enemy boost (total 49 per move)
  - All in all, totals to $6*26+18+14+4*49=388$ input dimensions.

- Reward thoughts:

  - +5 for fainting enemy pokemon.
  - -5 for fainting in self party pokemon.
  - +0.1 for % stat buff / enemy stat debuff
  - -0.1 for % stat debuff / enemy stat buff
  - +% health of enemy pokemon dealt
  - -% heath of party pokemon dealt

  To let the agent learn the basics of pokemon for the first few iterations

  - -20 for having > $\varepsilon$ probability for fainted pokemon.
  - -20 for dynamaxing but probability distribution for switching > $\varepsilon$.

- Output thoughts:
  - 6 dimension for pokemon and 4 for moveset = $10$ dimensions.
