import numpy as np
import fluent_data_loader as fdl
from scipy.optimize import minimize
from sklearn.cluster import KMeans

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
        self._action_preconditions = {}
        self._compute_action_preconditions()

    def _compute_action_preconditions(self):
        for action_label, (start_examples, df_examples) in self._action_data.items():
            action_var = np.var(start_examples, axis=0)
            x0 = np.random.standard_normal(len(action_var))

            def relevant_variance(w):
                return np.linalg.norm(w * action_var, ord=1)  # + 100 * np.linalg.norm(w, ord=1)

            cons = ({'type': 'ineq', 'fun': lambda x: -x + 1},  # -x + 1 > 0 => -x > -1 => x < 1
                    {'type': 'ineq', 'fun': lambda x: x},  # x > 0
                    {'type': 'ineq', 'fun': lambda x: np.linalg.norm(x, ord=1) - 3})
            res = minimize(relevant_variance, x0, method='SLSQP', constraints=cons)
            solved_x = res['x']
            solved_x[res['x'] < 1e-4] = 0
            print(action_label, solved_x, res['success'], res['message'])
            self._action_preconditions[action_label] = solved_x

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
        best_dists, best_dfs = [], []
        for action_label, (start_examples, df_examples) in self._action_data.items():
            action_w = self._action_preconditions[action_label]
            min_precondition_dist = float('inf')
            best_df = None
            for start_example_idx, start_example in enumerate(start_examples):
                precondition_dist = np.linalg.norm(action_w * (fluent - start_example), ord=2)
                if precondition_dist < min_precondition_dist:
                    min_precondition_dist = precondition_dist
                    best_df = df_examples[start_example_idx, :]
            best_dists.append([min_precondition_dist])
            best_dfs.append(best_df)
        best_dists = np.asarray(best_dists)
        km = KMeans(n_clusters=2).fit(best_dists)
        cluster_idx = 0 if km.cluster_centers_[0] < km.cluster_centers_[1] else 1

        for best_dist_idx, best_dist in enumerate(best_dists):
            if km.labels_[best_dist_idx] == cluster_idx:
                action_label = self._action_data.keys()[best_dist_idx]
                best_df = best_dfs[best_dist_idx]
                actions.append((action_label, best_df))

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
    values = loader.get_values()

    plan = HierarchicalPlan(action_data)
    action_sequence = plan.infer_path(start_fluent, end_fluent)
    print('Action sequence: {}'.format(action_sequence))
