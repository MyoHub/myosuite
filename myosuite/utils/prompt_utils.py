""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

"""
Utility script to help with information verbosity produced by RoboHive
To control verbosity set env variable ROBOHIVE_VERBOSITY=NONE(default)/INFO/WARN/ERROR/ONCE/ALWAYS
"""

from termcolor import cprint
import enum
import os


# Define verbosity levels
class Prompt(enum.IntEnum):
    """Prompt verbosity types"""
    NONE = 0    #  default (lowest priority)
    INFO = 1
    WARN = 2
    ERROR = 3
    ONCE = 4    # print only once
    ALWAYS = 5  # print always (highest priority)


# Prompt Cache (to track for Prompt.ONCE messages)
PROMPT_CACHE = []


# Infer verbose mode to be used
VERBOSE_MODE = os.getenv('ROBOHIVE_VERBOSITY')
if VERBOSE_MODE==None:
    VERBOSE_MODE = Prompt.WARN
else:
    VERBOSE_MODE = VERBOSE_MODE.upper()
    if VERBOSE_MODE == 'ALWAYS':
        VERBOSE_MODE = Prompt.ALWAYS
    elif VERBOSE_MODE == 'ERROR':
        VERBOSE_MODE = Prompt.ERROR
    elif VERBOSE_MODE == 'WARN':
        VERBOSE_MODE = Prompt.WARN
    elif VERBOSE_MODE == 'INFO':
        VERBOSE_MODE = Prompt.INFO
    elif VERBOSE_MODE == 'NONE':
        VERBOSE_MODE = Prompt.NONE
    else:
        raise TypeError("Unknown ROBOHIVE_VERBOSITY option")


# Programatically override the verbosity
def set_prompt_verbosity(verbose_mode:Prompt=Prompt.NONE):
    global VERBOSE_MODE
    VERBOSE_MODE = verbose_mode


# Print information respecting the verbosty mode
def prompt(data, color=None, on_color=None, flush=False, end="\n", type:Prompt=Prompt.INFO):

    global PROMPT_CACHE

    # Resolve if we need to print
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
    if type>=VERBOSE_MODE:
        if not isinstance(data, str):
            data = data.__str__()
        cprint(data, color=color, on_color=on_color, flush=flush, end=end)
