import numpy as np
import fluent_data_loader as fdl


class SearchState:
    def __init__(self, val, hist=[]):
        self._val = val
        self._hist = hist

    def get_val(self):
        return self._val

    def get_hist(self):
        return self._hist


class HierarchicalPlan:
    def __init__(self, action_data):
        self._action_data = action_data

    def infer_path(self, start_fluent, end_fluent, max_loop_count=1000):
        initial_state = SearchState(start_fluent)
        states = [initial_state]
        for i in range(max_loop_count):
            # pop best state
            min_cost_idx, cost = self._compute_heuristic_cost(states, end_fluent)
            state = states[min_cost_idx]
            del states[min_cost_idx]

            # expand best state
            fluent = state.get_val()
            actions = self._infer_actions(fluent)
            for action_label, fluent_change in actions:
                next_fluent = fluent + fluent_change
                hist_item = (cost, action_label)
                next_hist = state.get_hist() + [hist_item]
                next_state = SearchState(next_fluent, next_hist)
                states.append(next_state)
        min_cost_idx, cost = self._compute_heuristic_cost(states, end_fluent)
        return states[min_cost_idx].get_hist()

    def _infer_actions(self, fluent):
        """
        :param fluent: fluent vector
        :return: list of tuples (action_label, fluent_change)
        """
        actions = []
        for action_label, (start_examples, end_examples) in self._action_data.items():
            fluent_change = np.mean(end_examples - start_examples, axis=0)
            actions.append((action_label, fluent_change))
        return actions

    @staticmethod
    def _compute_heuristic_cost(states, end_fluent):
        min_cost_idx = None
        min_heuristic_cost = float('inf')
        for idx, state in enumerate(states):
            heuristic_cost = np.linalg.norm(state.get_val() - end_fluent)
            if heuristic_cost < min_heuristic_cost:
                min_heuristic_cost = heuristic_cost
                min_cost_idx = idx
        return min_cost_idx, min_heuristic_cost


if __name__ == '__main__':
    loader = fdl.DataLoader()
    action_data = loader.load_action_data()
    start_fluent, end_fluent = loader.get_goal(0)

    plan = HierarchicalPlan(action_data)
    action_sequence = plan.infer_path(start_fluent, end_fluent)
    print('Action sequence: {}'.format(action_sequence))
