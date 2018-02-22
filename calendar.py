"""
Track valid dates, number of days between them, etc.
"""

import re
import datetime, time
import numpy
from . import logging


def today():
    """Return system date in YYYYMMDD."""
    return time.strftime('%Y%m%d')


def decimalize(dates):
    """
    Convert '20020930' to 2002.75.
    """

    if numpy.isscalar(dates):
        date = str(dates)
        if not date: return numpy.nan
        return float(date[:4]) + (float(date[4:6]) - 1) / 12 + float(date[6:8]) / 31 / 12

    results = []
    for date in numpy.array(dates).flatten():
        date = str(date)
        if not date: results.append(numpy.nan)
        else: results.append(float(date[:4]) + (float(date[4:6]) - 1) / 12
                             + float(date[6:8]) / 31 / 12)
    return numpy.array(results).reshape(numpy.shape(dates))


def dt(dates):
    """
    Convert '20020930' to datetime.date(2002, 9, 30)
    """

    if numpy.isscalar(dates):
        date = str(dates)
        return datetime.date(int(date[:4]), int(date[4:6]), int(date[6:8]))

    results = []
    for date in numpy.array(dates).flatten():
        date = str(date)
        results.append(datetime.date(int(date[:4]), int(date[4:6]), int(date[6:8])))
    return numpy.array(results).reshape(numpy.shape(dates))

        
class Calendar(object):
    """
    The calendar object treats dates as 8 character YYYYMMDD strings.
    
    The dates are stored in two object attributes. An ordered numpy array,
    obj.dates. And a hash obj.date with the date as the key and the index of the
    date in obj.dates as the value.
    """

    date_expression = re.compile('^\d{8}$')

    def __init__(self, dates_file_or_list, start=None, finish=None,
                 select_tag=None, skip_tag=None, logger='warning'):
        """
        Initialize with dates in dates_file_or_list (one per line if file),
        as the valid dates of the calendar.

        start, finish:
        Limit calendar to dates between these dates.

        select_tag, skip_tag:
        Split on whitespace each line in dates_file_or_list, which is of the form
        '<date> <tag>', and load only dates with or without the given tag respectively.
        """
        
        if start: start = str(start)
        if finish: finish = str(finish)
        
        dates_list = dates_file_or_list
        if type(dates_list) == str:
            file_dates = open(dates_file_or_list)
            dates_list = file_dates.readlines()

        dates = []
        self.date = {}
        for count, date in enumerate(dates_list):
            tokens = date.split()
            date = tokens.pop(0)
            if skip_tag and tokens and tokens[0] == skip_tag: continue
            if select_tag and (not tokens or tokens[0] != select_tag): continue
            
            match = Calendar.date_expression.match(date)
            if not match:
                raise ValueError('Bad date in ' + dates_file + ': ' + date)
            if start and date < start: continue
            if finish and date > finish: break
            dates.append(date)
            self.date[date] = count
        self.dates = numpy.array(dates)
        
        if type(logger) == str:
            self.logger = logging.Logger(self.__class__.__name__, logger)
        else: self.logger = logger
        self.logger.debug('Loaded calendar')


    def _date_index(self, date):
        """Returns index of date in self.dates that is closest <= given date."""
        index = None
        date = str(date)
        if date in self.date:
            index = self.date[date]
        else:
            match = Calendar.date_expression.match(date)
            if not match: raise ValueError('Bad date: ' + date)
            index = numpy.searchsorted(self.dates, date) - 1
        if index is None: raise ValueError('Bad date for calendar: ' + date)
        return index


    def search_date(self, date, offset, clip_old=False, clip_new=False):
        """Returns date offset calendar-days away from given date."""
        index = self._date_index(date)
        new_index = index + offset
        if new_index < 0:
            if clip_old:
                self.logger.info('Clipping offset date to oldest date in calendar %s %d %s',
                                 date, offset, self.dates[0])
                return self.dates[0]
            else:
                raise IndexError('Date ' + date + ' offset ' + str(offset) +
                                 ' outside calendar range.')
        elif new_index >= len(self.dates):
            if clip_new:
                self.logger.info('Clipping offset date to newest date in calendar %s %d %s',
                                 date, offset, self.dates[-1])
                return self.dates[-1]
            else:
                raise IndexError('Date ' + date + ' offset ' + str(offset) +
                                 ' outside calendar range.')
        else:
            return self.dates[new_index]


    def next_date(self, date):
        """Returns next date in the calendar that is subsequent to given date."""
        return self.search_date(date, 1)


    def previous_date(self, date):
        """Returns previous date in the calendar that is earlier than given date."""
        return self.search_date(date, -1)


    def next_valid(self, date):
        """If given date is in calendar, return it; otherwise return closest
        subsequent date in the calendar."""
        date = str(date)
        if date in self.date: return date
        else: return self.search_date(date, 1)
        

    def previous_valid(self, date):
        """If given date is in calendar, return it; otherwise return closest
        earlier date in the calendar."""
        date = str(date)
        if date in self.date: return date
        else: return self.search_date(date, 0)


    def next_valid_close(self, date, time):
        """
        Convert date, time strings 'YYYYMMDD', 'HH:MM:SS' to 'YYYYMMDD', such that
        the resulting date is the next valid date-time 'YYYYMMDD 16:00:00'.
        """
        date = str(date)
        if date not in self.date: return self.next_valid(date)
        elif time > '15:59:59': return self.next_date(date)
        else: return date


    def days_between(self, date_older, date_newer):
        """Returns number of days between date_older and date_newer;
           returns a negative value if date_older is more recent."""
        return self._date_index(date_newer) - self._date_index(date_older)

    
    def dates_between(self, date_older, date_newer):
        """Returns dates >= date_older and < date_newer."""
        return self.dates[self.date[self.next_valid(date_older)]:
                          self.date[self.previous_date(date_newer)] + 1]

    
    def valid_dates_between(self, date_older, date_newer):
        """Returns dates >= date_older and <= date_newer."""
        return self.dates[self.date[self.next_valid(date_older)]:
                          self.date[self.next_date(date_newer)]]

    
    def next_year(self, date, include_current=False):
        """Returns next first-day-of-year in the calendar that is subsequent to
        given date. Set include_current flag if you want >= current date."""

        date = str(date)
        year, month, day = date[:4], date[4:6], date[6:]
        
        current_first_doy = self.next_valid(year + '0101')
        if date < current_first_doy or \
               (include_current and date == current_first_doy):
            return current_first_doy
        
        return self.next_valid(str(int(year)+1) + '0101')


    def next_valid_year(self, date):
        return self.next_year(date, include_current=True)

    
    def previous_year(self, date, include_current=False):
        """Returns previous first-day-of-year in the calendar that is earlier
        than given date. Set include_current flag if you want >= current date."""

        date = str(date)
        year, month, day = date[:4], date[4:6], date[6:]
        
        current_first_doy = self.next_valid(year + month + '0101')
        if date > current_first_doy or \
               (include_current and date == current_first_doy):
            return current_first_doy
        
        return self.next_valid(str(int(year)-1) + '0101')


    def previous_valid_year(self, date):
        return self.previous_year(date, include_current=True)


    def next_month(self, date, include_current=False):
        """Returns next first-day-of-month in the calendar that is subsequent to
        given date. Set include_current flag if you want >= current date."""

        date = str(date)
        year, month, day = date[:4], date[4:6], date[6:]
        
        current_first_dom = self.next_valid(year + month + '01')
        if date < current_first_dom or \
               (include_current and date == current_first_dom):
            return current_first_dom
        
        if month == '12':
            year = str(int(year) + 1)
            month = '01'
        else:
            month = str(int(month) + 1)
            if len(month) == 1: month = '0' + month
            
        return self.next_valid(year + month + '01')


    def next_valid_month(self, date):
        return self.next_month(date, include_current=True)

    
    def previous_month(self, date, include_current=False):
        """Returns previous first-day-of-month in the calendar that is earlier
        than given date. Set include_current flag if you want >= current date."""

        date = str(date)
        year, month, day = date[:4], date[4:6], date[6:]
        
        current_first_dom = self.next_valid(year + month + '01')
        if date > current_first_dom or \
               (include_current and date == current_first_dom):
            return current_first_dom
        
        if month == '01':
            year = str(int(year) - 1)
            month = '12'
        else:
            month = str(int(month) - 1)
            if len(month) == 1: month = '0' + month
            
        return self.next_valid(year + month + '01')


    def previous_valid_month(self, date):
        return self.previous_month(date, include_current=True)


    def next_quarter(self, date, include_current=False):
        """Returns next last-day-of-quarter in the calendar that is subsequent to
        given date. Set include_current flag if you want >= current date."""

        date = str(date)
        year = date[:4]
        quarters = [self.previous_valid(year + q)
                    for q in ('0331', '0630', '0930', '1231')]
        
        if include_current: index = numpy.searchsorted(quarters, date, side='left')
        else: index = numpy.searchsorted(quarters, date, side='right')
        
        if index < 4: return quarters[index]
        else: return self.previous_valid(str(int(year) + 1) + '0331')


    def next_valid_quarter(self, date):
        return self.next_quarter(date, include_current=True)

    
    def previous_quarter(self, date, include_current=False):
        """Returns next last-day-of-quarter in the calendar that is subsequent to
        given date. Set include_current flag if you want >= current date."""

        date = str(date)
        year = date[:4]
        quarters = [self.previous_valid(year + q)
                    for q in ('0331', '0630', '0930', '1231')]
        
        if include_current: index = numpy.searchsorted(quarters, date, side='right')
        else: index = numpy.searchsorted(quarters, date, side='left')
        
        if index > 0: return quarters[index - 1]
        else: return self.previous_valid(str(int(year) - 1) + '1231')


    def previous_valid_quarter(self, date):
        return self.previous_quarter(date, include_current=True)

