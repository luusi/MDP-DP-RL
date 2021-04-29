from typing import Sequence, Callable, Tuple, NoReturn
from abc import ABC, abstractmethod
from mdp_dp_rl.processes.mab_env import MabEnv
from numpy import ndarray, mean, vstack, cumsum, full, bincount
from mdp_dp_rl.utils.gen_utils import memoize


class MABBase(ABC):

    def __init__(
        self,
        mab: MabEnv,
        time_steps: int,
        num_episodes: int
    ) -> None:
        self.mab_funcs: Sequence[Callable[[], float]] = mab.arms_sampling_funcs
        self.num_arms: int = len(self.mab_funcs)
        self.time_steps: int = time_steps
        self.num_episodes: int = num_episodes

    @abstractmethod
    def get_episode_rewards_actions(self) -> Tuple[ndarray, ndarray]:
        pass

    @memoize
    def get_all_rewards_actions(self) -> Sequence[Tuple[ndarray, ndarray]]:
        return [self.get_episode_rewards_actions() for _ in range(self.num_episodes)]

    def get_rewards_matrix(self) -> ndarray:
        return vstack([x for x, _ in self.get_all_rewards_actions()])

    def get_actions_matrix(self) -> ndarray:
        return vstack([y for _, y in self.get_all_rewards_actions()])

    def get_expected_rewards(self) -> ndarray:
        return mean(self.get_rewards_matrix(), axis=0)

    def get_expected_cum_rewards(self) -> ndarray:
        return cumsum(self.get_expected_rewards())

    def get_expected_regret(self, best_mean) -> ndarray:
        return full(self.time_steps, best_mean) - self.get_expected_rewards()

    def get_expected_cum_regret(self, best_mean) -> ndarray:
        return cumsum(self.get_expected_regret(best_mean))

    def get_action_counts(self) -> ndarray:
        return vstack([bincount(ep, minlength=self.num_arms)
                       for ep in self.get_actions_matrix()])

    def get_expected_action_counts(self) -> ndarray:
        return mean(self.get_action_counts(), axis=0)

    def plot_exp_cum_regret_curve(self, best_mean) -> NoReturn:
        import matplotlib.pyplot as plt
        x_vals = range(1, self.time_steps + 1)
        plt.plot(self.get_expected_cum_regret(best_mean), "b", label="Exp Cum Regret")
        plt.xlabel("Time Steps", fontsize=20)
        plt.ylabel("Expected Cumulative Regret", fontsize=20)
        plt.title("Cumulative Regret Curve", fontsize=25)
        plt.xlim(xmin=x_vals[0], xmax=x_vals[-1])
        plt.ylim(ymin=0.0)
        # plt.xticks(x_vals)
        plt.grid(True)
        # plt.legend(loc='upper left')
        plt.show()


