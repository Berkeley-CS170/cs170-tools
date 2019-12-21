from .pipeline import *
from .util import levenshtein
import pandas as pd
import numpy as np
import os, os.path

class read_csv(Stage):

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

class save_csv(Stage):

    def process(self, ctx):
        source = self.arg('source')
        file = self.arg('file')
        dr = os.path.dirname(file)
        if not os.path.exists(dr):
            os.makedirs(dr)
        ctx[source].to_csv(file, index=False)

class create_assignments(Stage):

    """
    Create the assignments table.
    """

    def process(self, ctx):
        hws, hw_points, assgn = self.args('hws', 'hw_points', 'assgn')
        hw_points_each = hw_points / len(hws)
        assignments = []

        for hw in hws:
            assignments.append(dict(
                id='hw{}'.format(hw),
                type='hw',
                name='Homework {}'.format(hw),
                weight=hw_points_each
            ))

        def get_type(assgn_id):
            if assgn_id.startswith('mt') or assgn_id == 'final': return 'exam'
            elif assgn_id == 'proj': return 'project'
            else: raise ValueError()
        
        def get_name(assgn_id):
            if assgn_id.startswith('mt'): return 'Midterm {}'.format(assgn_id[2:])
            elif assgn_id == 'final': return 'Final Exam'
            elif assgn_id == 'proj': return 'Project'
            else: raise ValueError()
            

        for assgn_id in assgn:
            weight = assgn[assgn_id]
            assgn_type = get_type(assgn_id)
            assgn_name = get_name(assgn_id)
            assignments.append(dict(
                id=assgn_id,
                type=assgn_type,
                name=assgn_name,
                weight=weight
            ))
        
        ctx['assignments'] = pd.DataFrame(assignments)

        S = ctx['assignments']['weight'].sum()
        if S != 1.0:
            print('WARN: sum of assignment weights is {}, not 1.0'.format(S))

class match_students(Stage):

    """
    Ensures that all students on bcourses are in gradescope.
    If not, creates fake students so that grading can continue.
    Also computes what emails are closest to help matching easier.
    """

    def process(self, ctx):
        gradescope, bcourses = ctx['gradescope'], ctx['bcourses']

        # make sure both have string sids
        gradescope['SID'] = gradescope['SID'].astype('U')
        bcourses['Student ID'] = bcourses['Student ID'].astype('U')

        s = set(gradescope['SID'])

        def dist_to(value):
            def apply(row):
                return levenshtein(value, row['Email'])
            return apply

        def print_candidates(row):
            print('  ', row['Name'], row['Email'], row['SID'])

        mismatch = False

        def check_ids(row):
            nonlocal mismatch
            nonlocal gradescope
            if row['Student ID'] not in s:
                print('WARN: student "{}" <{}> {} on bcourses not found in gradescope'.format(row['Name'], row['Email Address'], row['Student ID']))

                # compute text distances to find candidates in gradescope
                email = row['Email Address']
                g = gradescope.copy()
                g['email_dists'] = g.apply(dist_to(email), axis=1)
                by_emails = g.sort_values('email_dists')
                by_emails = by_emails[by_emails['email_dists'] <= 3]
                if by_emails.shape[0] > 0:
                    print('HOWEVER: the following could be candidates')
                    by_emails.apply(print_candidates, axis=1)
                mismatch = True

                gradescope = gradescope.append({'Name': row['Name'] + ' (** MISSING)', 'Email': email, 'SID': row['Student ID']}, ignore_index=True)

        bcourses.apply(check_ids, axis=1)

        if mismatch:
            print('SUGGESTION: Match the missing students and add their sids to Gradescope, then re-download the Gradescope data')
            print('  (So the script can continue, fake students with 0s have been added to Gradescope)')

        ctx['gradescope'] = gradescope
        ctx['bcourses'] = bcourses

class populate_assignments(Stage):

    """
    Populates the assignment table with point values from gradescope.
    """

    def process(self, ctx):
        assgn = ctx['assignments']
        gradescope = ctx['gradescope']
        gradescope_assignments_explicit = self.arg_opt('gradescope_assignments', dict())
        point_values_explicit = self.arg_opt('point_values', dict())

        # take the first student, we only care about the max point values anyway
        eg_row = gradescope.iloc[0]
        
        def gradescope_assignments(row):
            assgn_id = row['id']
            if assgn_id in gradescope_assignments_explicit:
                return gradescope_assignments_explicit[assgn_id]
            else:
                return [row['name']]

        assgn['gradescope'] = assgn.apply(gradescope_assignments, axis=1)

        def points(row):
            assgn_id = row['id']
            gradescope_assignments = row['gradescope']
            if assgn_id in point_values_explicit:
                return point_values_explicit[assgn_id]
            else:
                points = list(set(
                    eg_row[g + ' - Max Points'] for g in gradescope_assignments
                ))
                if len(points) > 1:
                    print('WARN: assignment {} has multiple point values {}, picking {}'.format(
                        assgn_id, ', '.join(str(x) for x in points), points[0]
                        ))
                return points[0]

        ctx['assignments'] = assgn
        assgn['points'] = assgn.apply(points, axis=1)

class populate_grades(Stage):

    """
    Creates the 'grades' table, populating the assignments and weights for each student
    """

    _defaults = dict(source='gradescope', on='assignments', dest='grades')

    def process(self, ctx):
        bcourses = ctx['bcourses']
        gradescope = ctx['gradescope']
        assgn = ctx['assignments']

        grading_data = bcourses.merge(gradescope, how='left', left_on='Student ID', right_on='SID', suffixes=('', '_g'))

        def mk_grades(row):
            student = dict(
                name=row['Name'],
                sid=row['Student ID'],
                email=row['Email Address']
            )
            
            def add_grades(assgn_row):
                aid = assgn_row['id']
                point_values = list(
                    row[a_name] for a_name in assgn_row['gradescope']
                    if a_name in row and not np.isnan(row[a_name])
                )
                student[aid + '-points'] = max(point_values) if len(point_values) > 0 else np.nan
                student[aid + '-max'] = assgn_row['points']
                student[aid + '-weight'] = assgn_row['weight']
                if student[aid + '-max'] != 0.0:
                    student[aid + '-score'] = student[aid+'-points'] / assgn_row['points']
                else:
                    student[aid + '-score'] = 0.0

            assgn.apply(add_grades, axis=1)

            return pd.Series(student)

        grades = grading_data.apply(mk_grades, axis=1)
        ctx['grades'] = grades

class homework_floors(Stage):

    """
    
    """

    def process(self, ctx):

        def floors(row):
            pass
        pass

class homework_drops(Stage):

    """
    Perform homework drops.
    """

    def process(self, ctx):
        pass

class exam_drops(Stage):

    """
    Perform exam drops
    """

    def process(self, ctx):
        pass

class add_pt(Stage):

    """
    Add points from one source to another
    """

    def process(self, ctx):
        pass

class create_buckets(Stage):

    """
    Create buckets table according to the given guidelines.
    """

    def process(self, ctx):
        pass

class adjust_buckets(Stage):

    """
    Scan above and below to find more natural bucket locations to separate students.
    """

    def process(self, ctx):
        pass

class assign_letters(Stage):

    def process(self, ctx):
        pass

class render_reports(Stage):

    def process(self, ctx):
        pass
