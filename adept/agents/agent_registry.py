# Copyright (C) 2018 Heron Systems, Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
from adept.agents.agent_module import AgentModule


class AgentRegistry:
    """
    Keeps track of supported agents.
    """

    def __init__(self):
        self._agent_class_by_id = {}

        self.register_agent(self._load_actor_critic())
        self.register_agent(self._load_actor_critic_vtrace())
        self.register_dqn_agents()

    @staticmethod
    def _load_actor_critic():
        from adept.agents.actor_critic import ActorCritic
        return ActorCritic

    @staticmethod
    def _load_actor_critic_vtrace():
        from adept.agents.impala.actor_critic_vtrace import ActorCriticVtrace
        return ActorCriticVtrace

    def register_dqn_agents(self):
        from adept.agents.dqn import DQN, DDQN, OnlineDQN, OnlineDDQN, ActorLearnerDQN, ActorLearnerDDQN, QRDQN, OnlineQRDQN
        for dqn in [DQN, DDQN, OnlineDQN, OnlineDDQN, ActorLearnerDQN, ActorLearnerDDQN, QRDQN, OnlineQRDQN]:
            self.register_agent(dqn)

    def register_agent(self, agent_class):
        """
        Add your own agent class.

        :param agent_class: adept.agents.AgentModule. Your custom class.
        :return:
        """
        assert issubclass(agent_class, AgentModule)
        agent_class.check_args_implemented()
        self._agent_class_by_id[agent_class.__name__] = agent_class

    def lookup_agent(self, agent_id):
        """
        Get an agent class by id.

        :param agent_id: str
        :return: adept.agents.AgentModule
        """
        return self._agent_class_by_id[agent_id]

    def lookup_output_space(self, agent_id, action_space, args):
        """
        For a given agent_id, determine the shapes of the outputs.

        :param agent_id: str
        :param action_space:
        :return:
        """
        return self._agent_class_by_id[agent_id].output_space(action_space, args)
