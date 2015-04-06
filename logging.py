"""
Lightweight, picklable, implementation of logging module, with optional message history.

Copyright 2013 Deepak Subburam

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

from time import ctime
import sys


def timestamp():
    """How timestamps appear in log messages: HH:MM:SS"""
    return ctime()[11:19]


class Logger(object):
    """
    SYNOPSIS

    l = Logger('loggername', 'info', store=True) # store all messages
    l.debug('Processing date %d symbol %s', date, symbol)
    <no output>
    
    l.info('Finished processing date %d', date)
    prints --
    [INFO  loggername 09:45:10] Finished processing date 20120219.

    l.history contains:
    [['18:56:01', 10, 'Processing date 20120219 symbol IBM'],
    ['18:56:27', 20, 'Finished processing date 20120219']]

    ATTRIBUTES

    obj.name:
    Output is prefixed with this string.

    obj.level:
    What level of messgages to make visible (print).

    obj.store_history, obj.history:
    whether to store all calls made to this object and the text associated with them, in
    obj.history. If False, only the most recent call is stored.

    obj.critical_exit:
    do sys.exit(msg) rather than return(msg) when self.critical(msg) is called.
    """

    level_value = {'critical': 50,
                   'error': 40,
                   'warning': 30,
                   'info': 20,
                   'debug': 10,
                   'notset': 0}
    level_str = { 0: 'VERB', # abbreviation of 'VERBOSE'; more informative than 'NOTSET'
                 10: 'DEBUG',
                 20: 'INFO',
                 30: 'WARN',
                 40: 'ERROR',
                 50: 'CRIT'}


    def __init__(self, name, level, store=False, critical_exit=True, store_notset=False):
        """

        name:
        Prefix output with this string.

        level:
        one of the standard python logger module levels.

        store:
        store all calls made to this object and the text associated with them, in
        obj.history. If False, only the most recent call is stored.

        critical_exit:
        do sys.exit(msg) rather than return(msg) when self.critical(msg) is called.
        """
        
        self.name = name
        self.setLevel(level)
        self.history = []
        self.store_history = store
        self.store_history_notset = store_notset
        self.critical_exit = critical_exit

    def setLevel(self, level):
        if level not in Logger.level_value:
            raise KeyError('%s not a valid error level', level)
        
        self.level = level
        self.level_value = Logger.level_value[level]


    def _handle(self, level, text):

        stamp = timestamp()
        if self.level_value <= level:
            sys.stderr.write('[{:<5} {} {}] {}\n'.format(Logger.level_str[level],
                                                         self.name, stamp, text))

        if (level and self.store_history) or \
           (not level and self.store_history_notset):
            self.history.append([stamp, level, text])
        else: self.history = [[stamp, level, text]]
        return self.history[-1]
    

    def debug(self, text, *args):
        return self._handle(Logger.level_value['debug'], text.format(*args))

    def info(self, text, *args):
        return self._handle(Logger.level_value['info'], text.format(*args))

    def warning(self, text, *args):
        return self._handle(Logger.level_value['warning'], text.format(*args))

    def error(self, text, *args):
        return self._handle(Logger.level_value['error'], text.format(*args))

    def critical(self, text, *args):
        result = self._handle(Logger.level_value['critical'], text.format(*args))
        if self.critical_exit:
            sys.exit('[{:<5} {} {}] Exiting...'.format(Logger.level_str[result[1]],
                                                       self.name, result[0]))
        else: return result

    def notset(self, text, *args):
        return self._handle(Logger.level_value['notset'], text.format(*args))

