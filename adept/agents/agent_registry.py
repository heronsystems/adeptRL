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
from adept.agents.agent_plugin import AgentPlugin


class AgentRegistry:
    """
    Keeps track of supported agents.
    """

    def __init__(self):
        self._agent_class_by_id = {}


        self.register_agent('ActorCritic', self._load_actor_critic())
        self.register_agent('ActorCriticVtrace', self._load_actor_critic_vtrace())

    def _load_actor_critic(self):
        from adept.agents.actor_critic import ActorCritic
        return ActorCritic

    def _load_actor_critic_vtrace(self):
        from adept.agents.impala.actor_critic_vtrace import ActorCriticVtrace
        return ActorCriticVtrace

    def register_agent(self, agent_id, agent_class):
        """
        Add your own agent class.

        :param agent_id: str Name of your agent.
        :param agent_class: adept.agents.AgentPlugin. Your custom class.
        :return:
        """
        assert issubclass(agent_class, AgentPlugin)
        agent_class.check_defaults()
        self._agent_class_by_id[agent_id] = agent_class

    def lookup_agent(self, agent_id):
        """
        Get an agent class by id.

        :param agent_id: str
        :return: adept.agents.AgentPlugin
        """
        return self._agent_class_by_id[agent_id]
