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

    obj.store_history, obj.history, obj.previous:
    whether to store calls made to this object and the text associated with them,
    in obj.history. If False, only the most recent call is stored, in obj.previous.

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


    def __init__(self, name, level, store=False, logfile=None, printer=sys.stderr,
                 critical_exit=True, store_notset=False):
        """
        name:
        Prefix output with this string.

        level:
        one of the standard python logger module levels.

        store:
        store all calls made to this object and the text associated with them, in
        obj.history. If False, only the most recent call is stored, in obj.previous.

        logfile:
        write all messages to this filehandle, in addition to stderr.
        
        critical_exit:
        do sys.exit(msg) rather than return(msg) when self.critical(msg) is called.
        """
        
        self.name = name
        self.setLevel(level)
        self.history = []
        self.previous = None
        self.store_history = store
        self.logfile = logfile
        self.printer = printer
        self.store_history_notset = store_notset
        self.critical_exit = critical_exit


    def setLevel(self, level):
        if level not in Logger.level_value:
            raise KeyError('%s not a valid error level', level)
        
        self.level = level
        self.level_value = Logger.level_value[level]


    def _handle(self, level, text, status_line=False, store=None):
        """
        status_line:
        If True, print on same line, overwriting previous status output.
        
        store:
        By default, text is stored in self.history (if self.store_history+True)
        and printed to self.logfile (if self.logfile is not None) unless
        status_line=True, or level is 0 and store_history_notset is False.
        Specify store=True (False) for a definitive store (or not).
        """

        stamp = timestamp()
        out_str = '[{:<5} {} {}] {}'.format(Logger.level_str[level],
                                            self.name, stamp, text)
        
        if self.level_value <= level:
            if self.previous and self.previous[-1]:
                if status_line:
                    # Erase the previous line.
                    self.printer.write('\r' + ' '*len(self.previous[-2]) + '\r')
                else:
                    # Step to next line.
                    # (Preserve previous line's output onscreen.)
                    self.printer.write('\n')
            self.printer.write(out_str + ('' if status_line else '\n'))

        self.previous = [stamp, level, text, out_str, status_line]
        if store is not False:
            if store is True or level or self.store_history_notset:
                if self.logfile is not None:
                    self.logfile.write(out_str + '\n')
                if self.store_history:
                    self.history.append([stamp, level, text])
        return self.previous
    

    def debug(self, text, *args, **kwargs):
        return self._handle(Logger.level_value['debug'], text.format(*args), **kwargs)

    def info(self, text, *args, **kwargs):
        return self._handle(Logger.level_value['info'], text.format(*args), **kwargs)

    def warning(self, text, *args, **kwargs):
        return self._handle(Logger.level_value['warning'], text.format(*args), **kwargs)

    def error(self, text, *args, **kwargs):
        return self._handle(Logger.level_value['error'], text.format(*args), **kwargs)

    def critical(self, text, *args, **kwargs):
        result = self._handle(Logger.level_value['critical'], text.format(*args), **kwargs)
        if self.critical_exit:
            sys.exit('[{:<5} {} {}] Exiting...'.format(Logger.level_str[result[1]],
                                                       self.name, result[0]))
        else: return result

    def notset(self, text, *args, **kwargs):
        return self._handle(Logger.level_value['notset'], text.format(*args), **kwargs)

