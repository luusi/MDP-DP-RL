from typing import Sequence, Tuple, List
from mdp_dp_rl.processes.mab_env import MabEnv
from operator import itemgetter
from numpy import ndarray, empty, sqrt, log
from mdp_dp_rl.algorithms.mab.mab_base import MABBase


class UCB1(MABBase):

    def __init__(
        self,
        mab: MabEnv,
        time_steps: int,
        num_episodes: int,
        bounds_range: float,
        alpha: float
    ) -> None:
        if bounds_range < 0 or alpha <= 0:
            raise ValueError
        super().__init__(
            mab=mab,
            time_steps=time_steps,
            num_episodes=num_episodes
        )
        self.bounds_range: float = bounds_range
        self.alpha: float = alpha

    def get_episode_rewards_actions(self) -> Tuple[ndarray, ndarray]:
        ep_rewards: ndarray = empty(self.time_steps)
        ep_actions: ndarray = empty(self.time_steps, dtype=int)
        for i in range(self.num_arms):
            ep_rewards[i] = self.mab_funcs[i]()
            ep_actions[i] = i
        counts: List[int] = [1] * self.num_arms
        means: List[float] = [ep_rewards[j] for j in range(self.num_arms)]
        for i in range(self.num_arms, self.time_steps):
            ucbs: Sequence[float] = [means[j] + self.bounds_range *
                                     sqrt(0.5 * self.alpha * log(i) / counts[j])
                                     for j in range(self.num_arms)]
            action: int = max(enumerate(ucbs), key=itemgetter(1))[0]
            reward: float = self.mab_funcs[action]()
            counts[action] += 1
            means[action] += (reward - means[action]) / counts[action]
            ep_rewards[i] = reward
            ep_actions[i] = action
        return ep_rewards, ep_actions


if __name__ == '__main__':
    binomial_count = 10
    binomial_probs = [0.4, 0.8, 0.1, 0.5, 0.9, 0.2]
    binomial_params = [(binomial_count, p) for p in binomial_probs]
    mu_star = max(n * p for n, p in binomial_params)
    steps = 200
    episodes = 1000
    this_range = binomial_count
    this_alpha = 4.0

    me = MabEnv.get_binomial_mab_env(binomial_params)
    ucb1 = UCB1(
        mab=me,
        time_steps=steps,
        num_episodes=episodes,
        bounds_range=this_range,
        alpha=this_alpha
    )
    exp_cum_regret = ucb1.get_expected_cum_regret(mu_star)
    print(exp_cum_regret)

    exp_act_count = ucb1.get_expected_action_counts()
    print(exp_act_count)

    ucb1.plot_exp_cum_regret_curve(mu_star)



