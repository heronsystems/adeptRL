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
from adept.actor import ActorModule
from adept.agents.agent_module import AgentModule


class AgentRegistry:
    """
    Keeps track of supported agents.
    """

    def __init__(self):
        self._agent_class_by_id = {}
        self._actor_class_by_id = {}
        self._register_agents()
        self._register_actors()
        # self.register_agent(self._load_actor_critic())
        # self.register_agent(self._load_actor_critic_vtrace())

    def _register_agents(self):
        from adept.agents import AGENT_REG
        for agent in AGENT_REG:
            self.register_agent(agent)

    def _register_actors(self):
        from adept.actor import ACTOR_REG
        for actor in ACTOR_REG:
            self.register_actor(actor)

    def register_agent(self, agent_class):
        """
        Add your own agent class.

        :param agent_class: adept.agents.AgentModule. Your custom class.
        :return:
        """
        assert issubclass(agent_class, AgentModule)
        agent_class.check_args_implemented()
        self._agent_class_by_id[agent_class.__name__] = agent_class

    def register_actor(self, actor_class):
        """
        Add your own actor class.

        :param actor_class: adept.actor.ActorModule. Your custom class.
        :return:
        """
        assert issubclass(actor_class, ActorModule)
        actor_class.check_args_implemented()
        self._actor_class_by_id[actor_class.__name__] = actor_class

    def lookup_eval_actor(self, train_name):
        """
        Get the eval actor by training agent or actor name.

        :param train_name: Name of agent or actor class used for training
        :return: ActorModule
        """
        from adept.actor import ACTOR_EVAL_LOOKUP
        from adept.agents import agent_eval_lookup
        agent_lookup = agent_eval_lookup()
        if train_name in agent_lookup:
            return self._actor_class_by_id[agent_lookup[train_name]]
        elif train_name in ACTOR_EVAL_LOOKUP:
            return self._actor_class_by_id[ACTOR_EVAL_LOOKUP[train_name]]
        else:
            raise IndexError(f'Unknown training agent or actor: {train_name}')

    def lookup_agent(self, agent_id):
        """
        Get an agent class by id.

        :param agent_id: str
        :return: adept.agents.AgentModule
        """
        return self._agent_class_by_id[agent_id]

    def lookup_output_space(self, agent_id, action_space):
        """
        For a given agent_id, determine the shapes of the outputs.

        :param agent_id: str
        :param action_space:
        :return:
        """
        if agent_id in self._agent_class_by_id:
            return self._agent_class_by_id[agent_id].output_space(action_space)
        elif agent_id in self._actor_class_by_id:
            return self._actor_class_by_id[agent_id].output_space(action_space)
        else:
            raise IndexError(f'Actor or Agent not found: {agent_id}')
