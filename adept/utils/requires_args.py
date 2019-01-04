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


class RequiresArgs:
    args = None

    @classmethod
    def check_defaults(cls):
        if cls.args is None:
            raise NotImplementedError(
                'Subclass must define class attribute "args"'
            )

    @classmethod
    def prompt(cls):
        """
        Display defaults as JSON, prompt user for changes.

        :return: Dict[str, Any] Updated config dictionary.
        """
        if not cls.args:
            return cls.args

        user_input = input(
            '\n{} Defaults:\n{}\n'
            'Press ENTER to use defaults. Otherwise, '
            'modify JSON keys then press ENTER.\n'.format(
                cls.__name__,
                json.dumps(cls.args, indent=2, sort_keys=True)
            ) + 'Example: {"some_key": <new_value>, "another_key": '
                '<new_value>}\n'
        )

        # use defaults if no changes specified
        if user_input == '':
            return cls.args

        updates = json.loads(user_input)
        return {**cls.args, **updates}
