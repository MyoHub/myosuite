""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

"""
Utility script to help with information verbosity produced by RoboHive
To control verbosity set env variable ROBOHIVE_VERBOSITY=ALL/INFO/(WARN)/ERROR/ONCE/ALWAYS/SILENT
"""

from termcolor import cprint
import enum
import os


# Define verbosity levels
class Prompt(enum.IntEnum):
    """Prompt verbosity types"""
    ALL = 0     # print everything (lowest priority)
    INFO = 1    # useful info
    WARN = 2    # warnings (default)
    ERROR = 3   # errors
    ONCE = 4    # print: once and higher
    ALWAYS = 5  # print: only always (highest priority)
    SILENT = 6  # Supress all prints


# Prompt Cache (to track for Prompt.ONCE messages)
PROMPT_CACHE = []


# Infer verbose mode to be used
VERBOSE_MODE = os.getenv('ROBOHIVE_VERBOSITY')
if VERBOSE_MODE==None:
    VERBOSE_MODE = Prompt.WARN
else:
    VERBOSE_MODE = VERBOSE_MODE.upper()
    if VERBOSE_MODE == 'SILENT':
        VERBOSE_MODE = Prompt.SILENT
    elif VERBOSE_MODE == 'ALWAYS':
        VERBOSE_MODE = Prompt.ALWAYS
    elif VERBOSE_MODE == 'ERROR':
        VERBOSE_MODE = Prompt.ERROR
    elif VERBOSE_MODE == 'WARN':
        VERBOSE_MODE = Prompt.WARN
    elif VERBOSE_MODE == 'INFO':
        VERBOSE_MODE = Prompt.INFO
    elif VERBOSE_MODE == 'ALL':
        VERBOSE_MODE = Prompt.ALL
    else:
        raise TypeError("Unknown ROBOHIVE_VERBOSITY option")


# Programatically override the verbosity
def set_prompt_verbosity(verbose_mode:Prompt=Prompt.ALL):
    global VERBOSE_MODE
    VERBOSE_MODE = verbose_mode


# Print information respecting the verbosty mode
def prompt(data, color=None, on_color=None, flush=False, end="\n", type:Prompt=Prompt.INFO):

    global PROMPT_CACHE

    if type == Prompt.ONCE:
        data_hash = hash(data)
        if data_hash in PROMPT_CACHE:
            type = Prompt.INFO
        else:
            PROMPT_CACHE.append(data_hash)
            type = Prompt.ALWAYS

    # resolve print colors
    if on_color == None:
        if type==Prompt.WARN:
            color = "black"
            on_color = "on_yellow"
        elif type==Prompt.ERROR:
            on_color = "on_red"

    # resolve printing
    if VERBOSE_MODE == Prompt.SILENT:
        return
    elif type>=VERBOSE_MODE:
        if not isinstance(data, str):
            data = data.__str__()
        cprint(data, color=color, on_color=on_color, flush=flush, end=end)
