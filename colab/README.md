- every player has its own state
    - Its <code>state.id</code> is exactly the player's id


## What each script/notebook does
- <code>30_</code>:
    - input is ndarray of shape <code>(9,21,2)</code>
    - <code>channel0</code> is the map with positive value representing gold, negative values representing the amount which would be subtracted from the energy of a player when it steps in.
    - <code>channel1</code> is all zeros but at the agent's position, which marks its energy.
    - This model ignores bots/adversaries behaviour, concentrating on maximizing its own interest.
    - $$0.99 r = \frac{\texttt{episode}}{\texttt{n_episodes}}$$
    - $$0.99 r = \frac{episode}{nepisodes}$$

- <code>32_</code>:
    - input is almost the same as <code>30_</code>, except that this time it takes into consideration bots' position (and energy/or not).

- <code>34_</code>: The same as <code>30_</code> but with <code>4</code> actions only
- <code>35_</code>: The same as <code>30_</code> but with <code>5</code> actions only



## About <code>30_11_dDQN_light_tweakxz.py</code> where <code>x</code> $\in [0..9]$, <code>z</code> $\in [1..5]$
- <code>tweak01..05</code> are tweaks
- <code>tweak11..15</code> are the same as <code>tweak01..05</code> except that <code>buf_fill</code> has been changed longer
- <code>tweak21..25</code> are the same as <code>tweak01..05</code> except that <code>buf_fill</code> has been changed shorter
- <code>tweak31..35</code> are the same as <code>tweak01..05</code> except that model has been made complexer
- <code>tweak41..45</code> are the same as <code>tweak01..05</code> except that model has been made simpler




