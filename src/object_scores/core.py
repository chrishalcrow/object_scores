from pynapple import Tsd, TsGroup
import pandas as pd
import matplotlib.pyplot as plt

from .scores import _compute_ori, _compute_increase, _compute_information_content
from .plotting import _plot_ori, _plot_increase, _plot_information_content


class SingleSession:
    """Represents a single experimental session with:
        
    """
    
    def __init__(self, spikes: TsGroup, Px: Tsd, Py: Tsd, object_dict=None):
        
        self.spikes = spikes
        self.Px = Px
        self.Py = Py
        if object_dict is None:
            self.object_dict = {}
        else:
            self.object_dict = object_dict

        self.scores = pd.DataFrame(spikes.index, columns=['cluster_id'])
        self.params = {}

    def has_object(self):
        return len(self.object_dict) > 0
    

class SessionGroup:
    """A list of sessions, which together form an experiment"""

    def __init__(self, sessions: dict[str, SingleSession]):

        self.sessions = sessions
        self.params = {}

        spikes = list(sessions.values())[0].spikes
        self.scores = pd.DataFrame(spikes.index, columns=['cluster_id'])
        self.scores_info = {}

        self.computed_scores = []

    def _repr_html_(self):

        html_text = ""
        html_text += f"<strong>Experiment</strong>"
        for a, session_name in enumerate(self.sessions.keys()):
            if a != 0:
                html_text += "↓"
            html_text += f"<div style='border: 2px solid black; width: 100px'>{session_name}</div>"

        return html_text
    
    def get_computable_scores(self):

        objects = [len(session.object_dict) > 0 for session in self.sessions.values()]

        computable_scores = []

        computable_scores.append('information_score')

        if any(objects):
            computable_scores.append('ori')

        if objects[:2] == [False, True]:
            computable_scores.append('increase')

        return computable_scores

    def compute_ori(self, session_names = None, cluster_ids=None, mask_cm=18):

        score_name = 'ori'

        self.params[score_name] = {'mask_cm': mask_cm}
        
        ori_scores = {}
        ori_info = {}
        for session_name, session in self.sessions.items():

            if session_names is not None:
                if session_name not in session_names:
                    break

            for object_name, object_position in session.object_dict.items():

                spikes = session.spikes
                Px = session.Px
                Py = session.Py

                name = f'{score_name}_{session_name}_{object_name}'

                ori_scores[name], ori_info[name] = _compute_ori(spikes, Px, Py, object_position, cluster_ids, mask_cm)
                
                self.scores[name] = ori_scores[name].values()
                self.scores_info[name] = ori_info[name]

        self.computed_scores.append(score_name)

        return self.scores

    def compute_increase(self, cluster_ids=None, mask_cm=18):

        score_name = "increase"

        no_object_session_name, no_object_session = list(self.sessions.items())[0]
        object_session_name, object_session = list(self.sessions.items())[1]

        if cluster_ids is None:
            cluster_ids = object_session.spikes.index

        of_spikes = no_object_session.spikes
        of_Px = no_object_session.Px
        of_Py = no_object_session.Py

        obj_spikes = object_session.spikes
        obj_Px = object_session.Px
        obj_Py = object_session.Py
        object_position = list(object_session.object_dict.values())[0]
        
        self.params[score_name] = {'mask_cm': mask_cm}

        increase_scores = {}
        increase_info = {}

        for object_name, object_position in object_session.object_dict.items():

            name = f'{score_name}_{no_object_session_name}_{object_session_name}_{object_name}'

            self.params[name] = {'mask_cm': mask_cm}

            for cluster_id in cluster_ids:
                increase_scores[cluster_id], increase_info[cluster_id] =  _compute_increase(of_spikes, of_Px, of_Py, obj_spikes, obj_Px, obj_Py, object_position, cluster_id, mask_cm)

            self.scores[name] = increase_scores.values()
            self.scores_info[name] = increase_info
            
        self.computed_scores.append(score_name)

        return self.scores
    
    def compute_information_content(self, cluster_ids=None, bins=None):

        score_name = "information_content"
        self.params[score_name] = {'bins': bins}

        information_scores = {}

        for session_name, session in self.sessions.items():

            spikes = session.spikes
            Px = session.Px
            Py = session.Py

            name = f"{score_name}_{session_name}"

            information_scores[name] =  _compute_information_content(spikes, Px, Py, cluster_ids, bins=bins)

            self.scores[name] = information_scores[name].values()

        self.computed_scores.append(score_name)

        return self.scores


    def plot_increase(self, cluster_id, sigma=2):

        mask_cm = self.params['increase']['mask_cm']
        
        no_object_session = list(self.sessions.values())[0]
        object_session = list(self.sessions.values())[1]

        increase_scores = [score_name for score_name in list(self.scores.columns) if 'increase' in score_name]
        increase_score = increase_scores[0]
        _, no_object_session_name, object_session_name, object_name = increase_score.split('_')

        name = f'increase_{no_object_session_name}_{object_session_name}_{object_name}'
        scores_info = self.scores_info[name]

        no_object_session = self.sessions[no_object_session_name]
        object_session = self.sessions[object_session_name]

        of_spikes = no_object_session.spikes
        of_Px = no_object_session.Px
        of_Py = no_object_session.Py

        obj_spikes = object_session.spikes
        obj_Px = object_session.Px
        obj_Py = object_session.Py
        object_position = list(object_session.object_dict.values())[0]

        return  _plot_increase(scores_info, of_spikes, of_Px, of_Py, obj_spikes, obj_Px, obj_Py, object_position, cluster_id, mask_cm, sigma=sigma)
    
    def plot_ori(self, cluster_id, sigma=2):

        n_objects = 0
        n_sessions = 0
        for session_name, session in self.sessions.items():
            if session.has_object():
                n_sessions += 1
                n_objects = max(n_objects, len(session.object_dict))

        
        fig, axes = plt.subplots(nrows = n_sessions, ncols=2*n_objects)
        import numpy as np
        print(f"{np.shape(axes)=}")
        axes_list = axes.flatten()

        mask_cm = self.params['ori']['mask_cm']

        axis_counter = 0
        
        for session_name, session in self.sessions.items():

            if session.has_object():

                for object_name, object_position in session.object_dict.items():

                    ax1 = axes_list[axis_counter]
                    ax2 = axes_list[axis_counter + 1]
                    axis_counter += 2

                    spikes = session.spikes
                    Px = session.Px
                    Py = session.Py
                    object_position = list(session.object_dict.values())[0]
                    ori_score = self.scores.query(f'cluster_id == {cluster_id}')[f'ori_{session_name}_{object_name}'].values[0]
                    On, An = self.scores_info[f'ori_{session_name}_{object_name}'][cluster_id]

                    _plot_ori(spikes[cluster_id], Px, Py, object_position, ori_score, On, An, mask_cm, fig, ax1, ax2, session_name, sigma=sigma)

        fig.tight_layout()
        
        return fig


    def plot_information_content(self, cluster_id, sigma=2):
       
        fig, axes = plt.subplots(nrows = len(self.sessions))

        for ax, (session_name, session) in zip(axes, self.sessions.items()):

            ax.set_aspect(1)

            spikes = session.spikes
            Px = session.Px
            Py = session.Py
            information_score = self.scores.query(f'cluster_id == {cluster_id}')[f'information_content_{session_name}'].values[0]

            _plot_information_content(spikes[cluster_id], Px, Py, fig, ax, session_name, information_score, sigma=sigma)

        fig.tight_layout()
        
        return fig

