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
import json
import abc


class RequiresArgsMixin(metaclass=abc.ABCMeta):
    """
    This mixin makes it so that subclasses must implement an args class
    attribute. These arguments are parsed at runtime and the user is offered a
    chance to change any desired args. Classes the use this mixin must
    implement the from_args() class method. from_args() is essentially a
    secondary constructor.
    """

    args = None  # Dict[str, Any]

    @classmethod
    def check_args_implemented(cls):
        if cls.args is None:
            raise NotImplementedError(
                'Subclass must define class attribute "args"'
            )

    @classmethod
    def prompt(cls, provided=None):
        """
        Display defaults as JSON, prompt user for changes.

        :param provided: Dict[str, Any], Override default prompts.
        :return: Dict[str, Any] Updated config dictionary.
        """
        if provided is not None:
            overrides = {k: v for k, v in provided.items() if k in cls.args}
            args = {**cls.args, **overrides}
        else:
            args = cls.args
        return cls._prompt(cls.__name__, args)

    @staticmethod
    def _prompt(name, args):
        """
        Display defaults as JSON, prompt user for changes.

        :param name: str Name of class
        :param args: Dict[str, Any]
        :return: Dict[str, Any] Updated config dictionary.
        """
        if not args:
            return args

        user_input = input(
            '\n{} Defaults:\n{}\n'
            'Press ENTER to use defaults. Otherwise, '
            'modify JSON keys then press ENTER.\n'.format(
                name,
                json.dumps(args, indent=2, sort_keys=True)
            ) + 'Example: {"x": True, "gamma": 0.001}\n'
        )

        # use defaults if no changes specified
        if user_input == '':
            return args

        updates = json.loads(user_input)
        return {**args, **updates}

    @classmethod
    @abc.abstractmethod
    def from_args(cls, *argss, **kwargs):
        raise NotImplementedError
