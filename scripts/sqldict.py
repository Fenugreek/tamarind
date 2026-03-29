"""Examine a sqlitedict file from the command line."""
import os, sys, argparse
from importlib import import_module
from sqlitedict import SqliteDict
import tamarind.logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sqlfname', metavar='<filename>', help='Sqlitedict file.')
    parser.add_argument('--tables', action='store_true', help='List tables in the sqldict.')
    parser.add_argument('--table', metavar='<tablename>', help='Operate on this table.')
    parser.add_argument('--init', metavar='<module.method>',
                        help='import this module and use that method to initialize the sqldict.')

    parser.add_argument('--keys', action='store_true', help='List keys in the table.')
    parser.add_argument('--values', action='store_true', help='List values in the table.')
    parser.add_argument('--items', action='store_true', help='List key value pairs in the table.')

    parser.add_argument('--log', metavar='<level> [logfile=<filename>]', nargs='+',
                        default=['info'], help='Output log messages here.')
    args = parser.parse_args()
    logger = tamarind.logging.Logger.from_opt(args.log, name=os.path.basename(__file__))

    if args.tables:
        for table in SqliteDict.get_tablenames(args.sqlfname):
            print(table)
        sys.exit()

    if not args.table:
        sys.exit('Need tablename to operate.')

    if args.init:
        init_tokens = args.init.split('.')
        module = import_module('.'.join(init_tokens[:-1]))
        sql = getattr(module, init_tokens[-1])(args.sqlfname, args.table)
    else:
        sql = SqliteDict(args.sqlfname, args.table)

    if args.keys:
        for key in sql.keys(): print(key)
    if args.values:
        for value in sql.values(): print(key)
    if args.items:
        for key in sql.keys():
            print(key)
            print(sql[key], '\n\n')
              
