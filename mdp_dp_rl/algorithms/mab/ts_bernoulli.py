from typing import Sequence, Tuple, List
from mdp_dp_rl.processes.mab_env import MabEnv
from operator import itemgetter
from numpy import ndarray, empty
from numpy.random import beta
from mdp_dp_rl.algorithms.mab.mab_base import MABBase


class ThompsonSamplingBernoulli(MABBase):

    def __init__(
        self,
        mab: MabEnv,
        time_steps: int,
        num_episodes: int
    ) -> None:
        super().__init__(
            mab=mab,
            time_steps=time_steps,
            num_episodes=num_episodes
        )

    def get_episode_rewards_actions(self) -> Tuple[ndarray, ndarray]:
        ep_rewards: ndarray = empty(self.time_steps)
        ep_actions: ndarray = empty(self.time_steps, dtype=int)
        bayes: List[Tuple[int, int]] = [(1, 1)] * self.num_arms

        for i in range(self.time_steps):
            mean_draws: Sequence[float] = [beta(a, b, 1)[0] for a, b in bayes]
            action: int = max(enumerate(mean_draws), key=itemgetter(1))[0]
            reward: float = self.mab_funcs[action]()
            a, b = bayes[action]
            bayes[action] = (a + int(reward), b + int(1 - reward))
            ep_rewards[i] = reward
            ep_actions[i] = action
        return ep_rewards, ep_actions


if __name__ == '__main__':
    probs_data = [0.2, 0.4, 0.8, 0.5, 0.1, 0.9]
    mu_star = max(probs_data)
    steps = 200
    episodes = 1000

    me = MabEnv.get_bernoulli_mab_env(probs_data)
    ucb1 = ThompsonSamplingBernoulli(
        mab=me,
        time_steps=steps,
        num_episodes=episodes
    )
    exp_cum_regret = ucb1.get_expected_cum_regret(mu_star)
    print(exp_cum_regret)

    exp_act_count = ucb1.get_expected_action_counts()
    print(exp_act_count)

    ucb1.plot_exp_cum_regret_curve(mu_star)
