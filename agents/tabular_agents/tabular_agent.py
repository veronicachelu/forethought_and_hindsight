import abc
import dm_env

class TabularAgent(abc.ABC):
    """An agent consists of a policy and an update rule."""
    def __init__(self):
        self.episode = 1
        self.total_steps = 1
        self.writer = None

    def policy(self,
               timestep: dm_env.TimeStep,
               eval: bool
               ) -> int:
        """A policy takes in a timestep and returns an action."""

    @abc.abstractmethod
    def value_update(
            self,
            timestep: dm_env.TimeStep,
            action: int,
            new_timestep: dm_env.TimeStep,
    ) -> None:
        """Updates the agent given a transition."""

    @abc.abstractmethod
    def model_based_train(self) -> bool:
        pass

    @abc.abstractmethod
    def model_free_train(self) -> bool:
        pass

    @abc.abstractmethod
    def save_transition(
            self,
            timestep: dm_env.TimeStep,
            action: int,
            new_timestep: dm_env.TimeStep,
    ) -> None:
        pass

    @abc.abstractmethod
    def model_update(
            self,
            timestep: dm_env.TimeStep,
            action: int,
            new_timestep: dm_env.TimeStep,
    ) -> None:
        pass

    @abc.abstractmethod
    def planning_update(
            self
    ) -> None:
        pass

    def save_model(self
    ) -> None:
        pass

    def load_model(self
    ) -> None:
        pass