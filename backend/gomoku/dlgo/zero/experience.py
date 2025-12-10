import numpy as np

__all__ = [
    'ZeroExperienceCollector',
    'ZeroExperienceBuffer',
    'combine_experience',
    'load_experience',
]

class ZeroExperienceCollector:
    def __init__(self):
        self.states = []
        self.visit_counts = []
        self.rewards = []
        self._current_episode_states = []
        self._current_episode_visit_counts = []

    def begin_episode(self):
        self._current_episode_states = []
        self._current_episode_visit_counts = []

    def record_decision(self, state, visit_counts):
        self._current_episode_states.append(state)
        self._current_episode_visit_counts.append(visit_counts)

    def complete_episode(self, reward):
        num_states = len(self._current_episode_states)
        self.states += self._current_episode_states
        self.visit_counts += self._current_episode_visit_counts
        self.rewards += [reward for _ in range(num_states)]

        self._current_episode_states = []
        self._current_episode_visit_counts = []

class ZeroExperienceBuffer:
    def __init__(self, states, visit_counts, rewards):
        self.states = np.array(states, dtype=object)
        self.visit_counts = np.array(visit_counts, dtype=object)
        self.rewards = np.array(rewards, dtype=object)

    def serialize(self, h5file):
        h5file.create_group('experience')
        h5file['experience'].create_dataset('states', data=self.states)
        h5file['experience'].create_dataset('visit_counts', data=self.visit_counts)
        h5file['experience'].create_dataset('rewards', data=self.rewards)

def combine_experience(collectors):
    combined_states = np.concatenate([np.array(c.states, dtype=object) for c in collectors])
    combined_visit_counts = []
    for c in collectors:
        for vc in c.visit_counts:
            combined_visit_counts.append(vc)
    combined_rewards = np.concatenate([np.array(c.rewards, dtype=object) for c in collectors])

    return ZeroExperienceBuffer(
        combined_states,
        combined_visit_counts,
        combined_rewards
    )

def load_experience(h5file):
    return ZeroExperienceBuffer(
        states=np.array(h5file['experience']['states'], dtype=object),
        visit_counts=np.array(h5file['experience']['visit_counts'], dtype=object),
        rewards=np.array(h5file['experience']['rewards'], dtype=object)
    )
