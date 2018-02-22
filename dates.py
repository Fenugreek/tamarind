"""
Some utilities for date manipulation.

Copyright 2013 Deepak Subburam

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import datetime
import numpy

def dt(dates, call=None):
    """
    Convert '20020930' to datetime.date(2002, 9, 30).

    If call is not None, after conversion, the date object's 'call'() method
    output is returned.
    """

    if numpy.isscalar(dates):
        date = str(dates)
        date = datetime.date(int(date[:4]), int(date[4:6]), int(date[6:8]))
        return getattr(date, call)() if call else date
        
    results = []
    for date in numpy.array(dates).flatten():
        date = str(date)
        date = datetime.date(int(date[:4]), int(date[4:6]), int(date[6:8]))
        results.append(getattr(date, call)() if call else date)
    return numpy.array(results).reshape(numpy.shape(dates))

