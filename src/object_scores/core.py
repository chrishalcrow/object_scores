from pynapple import Tsd, TsGroup
import pandas as pd
import matplotlib.pyplot as plt

from .scores import _compute_ori, _compute_increase, _compute_information_content
from .plotting import _plot_ori, _plot_increase, _plot_information_content


class SingleSession:
    """Represents a single experimental session with:
        
    """
    
    def __init__(self, spikes: TsGroup, Px: Tsd, Py: Tsd, object_position=None):
        
        self.spikes = spikes
        self.Px = Px
        self.Py = Py
        self.object_position=object_position

        self.scores = pd.DataFrame(spikes.index, columns=['cluster_id'])
        self.params = {}

    def has_object(self):
        return self.object_position is not None
    

class SessionGroup:
    """A list of sessions, which together form an experiment"""

    def __init__(self, sessions: dict[str, SingleSession]):

        self.sessions = sessions
        self.params = {}

        spikes = list(sessions.values())[0].spikes
        self.scores = pd.DataFrame(spikes.index, columns=['cluster_id'])
        self.scores_info = {}

    def _repr_html_(self):

        html_text = ""
        html_text += f"<strong>Experiment</strong>"
        for a, session_name in enumerate(self.sessions.keys()):
            if a != 0:
                html_text += "↓"
            html_text += f"<div style='border: 2px solid black; width: 100px'>{session_name}</div>"

        return html_text
    
    def get_computable_scores(self):

        objects = [session.object_position is not None for session in self.sessions.values()]

        computable_scores = []

        computable_scores.append('information_score')

        if any(objects):
            computable_scores.append('ori')

        if objects[:2] == [False, True]:
            computable_scores.append('increase')

        return computable_scores

    def compute_ori(self, session_names = None, cluster_ids=None, mask_cm=18):

        self.params['ori'] = {'mask_cm': mask_cm}
        
        ori_scores = {}
        ori_info = {}
        for session_name, session in self.sessions.items():

            if session_names is not None:
                if session_name not in session_names:
                    break

            if session.object_position is not None:

                spikes = session.spikes
                Px = session.Px
                Py = session.Py
                object_position = session.object_position

                ori_scores[f'ori_{session_name}'], ori_info[f'ori_{session_name}'] = _compute_ori(spikes, Px, Py, object_position, cluster_ids, mask_cm)
                
                if cluster_ids is None:
                    self.scores[f'ori_{session_name}'] = ori_scores[f'ori_{session_name}'].values()
                    self.scores_info[f'ori_{session_name}'] = ori_info[f'ori_{session_name}']

        return ori_scores

    def compute_increase(self, cluster_ids=None, mask_cm=18):

        no_object_session = list(self.sessions.values())[0]
        object_session = list(self.sessions.values())[1]

        if cluster_ids is None:
            cluster_ids = object_session.spikes.index

        of_spikes = no_object_session.spikes
        of_Px = no_object_session.Px
        of_Py = no_object_session.Py

        obj_spikes = object_session.spikes
        obj_Px = object_session.Px
        obj_Py = object_session.Py
        object_position = object_session.object_position
        
        self.params['increase'] = {'mask_cm': mask_cm}

        increase_scores = {}
        increase_info = {}
        for cluster_id in cluster_ids:
            increase_scores[cluster_id], increase_info[cluster_id] =  _compute_increase(of_spikes, of_Px, of_Py, obj_spikes, obj_Px, obj_Py, object_position, cluster_id, mask_cm)

        if cluster_ids is None:
            self.scores['increase'] = increase_scores.values()

        return increase_scores, increase_info
    
    def compute_information_content(self, cluster_ids=None, bins=None):

        information_scores = {}
        for session_name, session in self.sessions.items():

            spikes = session.spikes
            Px = session.Px
            Py = session.Py

            self.params['information_content'] = {'bins': bins}

            information_scores[f'information_content_{session_name}'] =  _compute_information_content(spikes, Px, Py, cluster_ids, bins=bins)

            if cluster_ids is None:
                self.scores[f'information_content_{session_name}'] = information_scores[f'information_content_{session_name}'].values()

        return information_scores


    def plot_increase(self, cluster_id, sigma=2):

        mask_cm = self.params['increase']['mask_cm']
        
        no_object_session = list(self.sessions.values())[0]
        object_session = list(self.sessions.values())[1]

        scores, scores_info = self.compute_increase([cluster_id])

        of_spikes = no_object_session.spikes
        of_Px = no_object_session.Px
        of_Py = no_object_session.Py

        obj_spikes = object_session.spikes
        obj_Px = object_session.Px
        obj_Py = object_session.Py
        object_position = object_session.object_position

        return  _plot_increase(scores_info, of_spikes, of_Px, of_Py, obj_spikes, obj_Px, obj_Py, object_position, cluster_id, mask_cm, sigma=sigma)
    
    def plot_ori(self, cluster_id, sigma=2):

        scored_sessions = {session_name: session for session_name, session in self.sessions.items() if self.scores.get(f'ori_{session_name}') is not None}
        
        fig, axes = plt.subplots(nrows = len(scored_sessions), ncols=2, figsize=(8, 4*len(scored_sessions)))

        mask_cm = self.params['ori']['mask_cm']

        for axs, (session_name, session) in zip(axes, scored_sessions.items()):

            spikes = session.spikes
            Px = session.Px
            Py = session.Py
            object_position = session.object_position
            ori_score = self.scores[f'ori_{session_name}'][cluster_id]
            On, An = self.scores_info[f'ori_{session_name}'][cluster_id]

            _plot_ori(spikes[cluster_id], Px, Py, object_position, ori_score, On, An, mask_cm, fig, axs, session_name, sigma=sigma)

        fig.tight_layout()
        
        return fig


    def plot_information_content(self, cluster_id, sigma=2):
       
        fig, axes = plt.subplots(nrows = len(self.sessions), figsize=(8, 4*len(self.sessions)))

        for ax, (session_name, session) in zip(axes, self.sessions.items()):

            spikes = session.spikes
            Px = session.Px
            Py = session.Py
            information_score = self.scores.query(f'cluster_id == {cluster_id}')[f'information_content_{session_name}'][cluster_id]

            _plot_information_content(spikes[cluster_id], Px, Py, fig, ax, session_name, information_score, sigma=sigma)

        fig.tight_layout()
        
        return fig

