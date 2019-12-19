from .pipeline import *
from .util import levenshtein
import pandas as pd
import numpy as np

class read_csv(Stage):

    _name = "read_csv"

    def process(self, ctx):
        dest = self.arg('dest')
        file = self.arg('file')

        cur_inputs = ctx.inputs()
        # only load the df if it is not already supplied.
        # this lets us rerun the pipeline!
        if dest not in cur_inputs:
            df = pd.read_csv(file)
            cur_inputs[dest] = df
        else:
            df = cur_inputs[dest]

        if 'stringify' in self.params:
            df = df.copy(deep=True)

        
        ctx[dest] = df
        
class create_homeworks(Stage):

    _name = "create_homeworks"
    _defaults = dict(on='assignments')

    def process(self, ctx):
        pass

class create_assignments(Stage):

    _name = "create_assignments"
    _defaults = dict(on='assignments')

    def process(self, ctx):
        pass

class populate_assignment_points(Stage):

    _name = "populate_assignment_points"
    _defaults = dict(source='gradescope', on='assignments')

    def process(self, ctx):
        pass

class match_students(Stage):

    _name = "match_students"

    def process(self, ctx):
        gradescope, bcourses = ctx['gradescope'], ctx['bcourses']

        # make sure both have string sids
        gradescope['SID'] = gradescope['SID'].astype('<U')
        bcourses['Student ID'] = bcourses['Student ID'].astype('<U')

        s = set(gradescope['SID'])

        def dist_to(value):
            def apply(row):
                return levenshtein(value, row['Email'])
            return apply

        def print_candidates(thres):
            def apply(row):
                if row['email_dists'] <= thres:
                    print('  ', row['Name'], row['Email'], row['SID'])
            return apply

        mismatch = False

        def check_ids(row):
            nonlocal mismatch
            if row['Student ID'] not in s:
                print('WARN: student "{}" <{}> {} on bcourses not found in gradescope'.format(row['Name'], row['Email Address'], row['Student ID']))

                # compute text distances to find candidates in gradescope
                email = row['Email Address']
                g = gradescope.copy()
                g['email_dists'] = g.apply(dist_to(email), axis=1)
                by_emails = g.sort_values('email_dists')
                print('HOWEVER: the following could be candidates')
                by_emails[:5].apply(print_candidates(3), axis=1)
                mismatch = True

        bcourses.apply(check_ids, axis=1)

        if mismatch:
            print('SUGGESTION: Match the missing students and add their sids to Gradescope, then re-download the Gradescope data')

        ctx['gradescope'] = gradescope
        ctx['bcourses'] = bcourses
