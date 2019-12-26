from .pipeline import *
from .util import levenshtein
import pandas as pd
import numpy as np
import scipy.stats
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
            stringify = self.params['stringify']
            cols = [stringify] if isinstance(stringify, str) else stringify
            for col in cols:
                df[col] = df[col].astype('U')
        
        ctx[dest] = df

class save_csv(Stage):

    def process(self, ctx):
        source = self.arg('source')
        file = self.arg('file')
        dr = os.path.dirname(file)
        if not os.path.exists(dr):
            os.makedirs(dr)
        ctx[source].to_csv(file, index=False)

class rename(Stage):

    def process(self, ctx):
        table = self.arg('table')
        cols = self.arg('cols')
        ctx[table] = ctx[table].rename(columns=cols)

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
        gradescope['sid'] = gradescope['sid'].astype('U')
        bcourses['Student ID'] = bcourses['Student ID'].astype('U')

        # make sure unique gradescope SIDs
        uniq_sid, counts = np.unique(gradescope['sid'], return_counts=True)
        duplicates = uniq_sid[counts > 1]
        counts_dup = counts[counts > 1]
        has_duplicate = False
        for i in range(len(counts_dup)):
            if duplicates[i] != 'nan':
                has_duplicate = True
                print('WARN: student id {} appears {} times in gradescope'.format(duplicates[i], counts_dup[i]))
        if has_duplicate:
            print('SUGGESTION: delete the unecessary duplicates from Gradescope, then re-download the Gradescope data')

        s = set(gradescope['sid'])

        def dist_to(value):
            def apply(row):
                return levenshtein(value, row['Email'])
            return apply

        def print_candidates(row):
            print('  ', row['Name'], row['Email'], row['sid'])

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

                gradescope = gradescope.append({'Name': row['Name'] + ' (** MISSING)', 'Email': email, 'sid': row['Student ID']}, ignore_index=True)

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

def get_score(points, max_points):
    if max_points != 0.0:
        return points / max_points
    else:
        return 0.0

class populate_grades(Stage):

    """
    Creates the 'grades' table, populating the assignments and weights for each student
    """

    _defaults = dict(source='gradescope', on='assignments', dest='grades')

    def process(self, ctx):
        bcourses = ctx['bcourses'].drop_duplicates(subset='Student ID')
        gradescope = ctx['gradescope'].drop_duplicates(subset='sid')
        assgn = ctx['assignments']

        grading_data = bcourses.merge(gradescope, how='left', left_on='Student ID', right_on='sid', suffixes=('', '_g'))

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
                student[aid + '-score'] = get_score(student[aid + '-points'], student[aid + '-max'])

            assgn.apply(add_grades, axis=1)

            return pd.Series(student)

        grades = grading_data.apply(mk_grades, axis=1)
        ctx['grades'] = grades

class homework_floors(Stage):

    """
    Create a floor on the homeworks.

    Exceptions processed per-student: 
        set_hw_floor (set the homework floor for the student globally)
        parameters: hw_floor 
    """

    def process(self, ctx):
        default = self.arg('default')
        assgn = ctx['assignments']
        hws = assgn[assgn['type'] == 'hw']
        grades = ctx['grades']

        floor_exceptions = dict()

        def find_exceptions(row):
            sid = row['sid']
            if row['type'] == 'set_hw_floor':
                floor_exceptions[sid] = row['hw_floor']

        if 'student_exceptions' in ctx:
            ctx['student_exceptions'].apply(find_exceptions, axis=1)

        def floors(row):
            sid = row['sid']
            floor = floor_exceptions.get(sid, default)
            hw_grades = dict()

            def update_score(hw):
                hw_id = hw['id']
                score = row[hw_id + '-score']
                if np.isnan(score):
                    hw_grades[hw_id + '-score'] = np.nan
                elif score < floor:
                    hw_grades[hw_id + '-score'] = score / floor
                else:
                    hw_grades[hw_id + '-score'] = 1.0
            
            hws.apply(update_score, axis=1)
            return pd.Series(hw_grades)

        new_hw_grades = grades.apply(floors, axis=1)
        grades.update(new_hw_grades)

class homework_drops(Stage):

    """
    Perform homework drops.

    Exceptions processed per-student: 
        set_num_hw_drops (set the number of homework drops for the student)
        parameters: num_hw_drops
    """

    def process(self, ctx):

        def hw_drops(row):
            pass


        pass

class exam_drops(Stage):

    """
    Perform exam drops. 

    Fill in using percentile, among students that are counted in the curve.
    """

    def process(self, ctx):
        assgn_exceptions = ctx['assignment_exceptions']
        grades = ctx['grades']

        # make a copy so as we edit grades, it doesn't change the curve
        grades_ref = ctx['grades'].copy(deep=True)

        drops = dict()

        def collect_drops(row):
             if row['type'] == 'drop_and_fill':
                sid = row['sid']
                if sid not in drops:
                    drops[sid] = []
                
                assgn_to_replace = row['assignment']
                assgn_fill_from = row['fill_from']

                drops[sid].append((assgn_to_replace, assgn_fill_from))

        assgn_exceptions.apply(collect_drops, axis=1)
        
        def drop_exam(row):
            changes = dict()
            sid = row['sid']

            if sid in drops:

                grades_row = grades[grades['sid'] == sid].iloc[0]

                for assgn_to_replace, assgn_fill_from in drops[sid]:
                    # take the curves
                    assgn_fill_curve = grades_ref[grades_ref['in-curve'] == True][assgn_fill_from + '-score'].to_numpy()
                    assgn_to_curve = grades_ref[grades_ref['in-curve'] == True][assgn_to_replace + '-score'].to_numpy()
                    assgn_fill_curve = assgn_fill_curve[~np.isnan(assgn_fill_curve)]
                    assgn_to_curve = assgn_to_curve[~np.isnan(assgn_to_curve)]

                    # translate the percentile rank
                    person_score = grades_row[assgn_fill_from + '-score']
                    if not np.isnan(person_score):
                        percentile = scipy.stats.percentileofscore(assgn_fill_curve, person_score)
                        translated_score = np.percentile(assgn_to_curve, percentile)
                        translated_points = translated_score * grades_row[assgn_to_replace + '-max']
                        changes[assgn_to_replace + '-points'] = translated_points
                        changes[assgn_to_replace + '-score'] = translated_score
                    else: 
                        S = 'WARN: student {name} {sid} doesn\'t have a score on {fill_from},' + \
                            ' so they will not have one on {to_replace} either despite drop'
                        print(S.format(
                            name=grades_row['name'],
                            sid=sid,
                            fill_from=assgn_fill_from,
                            to_replace=assgn_to_replace
                        ))

            return pd.Series(changes)

        changes = grades.apply(drop_exam, axis=1)
        changes.to_csv('changes.csv')
        grades.update(changes)

class add_pt(Stage):

    """
    Add points from one source to another
    """

    def process(self, ctx):
        aid, source = self.args('to', 'source')
        assignments = ctx['assignments']
        grades = ctx['grades']
     
        tbl = ctx[source[0]].copy(deep=True).drop_duplicates(subset='sid')
        use_existence = source[1] == '--existence'
        value = self.arg('value') if use_existence else np.nan
        col = '' if use_existence else source[1] + '_from'
        tbl.columns = tbl.columns.map(lambda c: str(c) + '_from')
        
        join_with_other = grades.merge(tbl, how='left', left_on='sid', right_on='sid_from', suffixes=(False, False))

        def add_point(row):
            aid_max = row[aid + '-max']
            aid_cur_points = row[aid + '-points']
            new_points = aid_cur_points

            if use_existence: 
                if not pd.isna(row['sid_from']):
                    new_points = aid_cur_points + value
            else:
                if not np.isnan(row[col]):
                    new_points = np.nan_to_num(aid_cur_points) + row[col]

            new_score = get_score(new_points, aid_max)
            return pd.Series({
                # debugging info
                # 'sid': row['sid'],
                # 'sid_from': row['sid_from'],
                # 'eq': '' if row['sid'] == row['sid_from'] else 'NOTEQ',
                # aid+'-pointsold': aid_cur_points,
                aid+'-points': new_points,
                aid+'-score': new_score
            })
        
        new_scores = join_with_other.apply(add_point, axis=1)
        grades.update(new_scores)

class compute_scores(Stage):

    def process(self, ctx):

        assignments = ctx['assignments']
        grades = ctx['grades']

        hw_weight = assignments[assignments['type'] == 'hw']['weight'].sum()
        
        def compute_for_student(row):
            score = 0
            hw_score = 0

            def per_assignment(assgn):
                nonlocal score, hw_score
                aid = assgn['id']
                if not np.isnan(row[aid + '-score']):
                    assgn_score = row[aid + '-score'] * row[aid + '-weight']
                    if aid.startswith('hw'):
                        hw_score += assgn_score
                    score += assgn_score

            assignments.apply(per_assignment, axis=1)
            
            return pd.Series({
                'total-score': score,
                'hw-score': hw_score / hw_weight
            })
        
        total_and_hw = grades.apply(compute_for_student, axis=1)
        grades = pd.concat([grades, total_and_hw], axis=1)
        grades = grades.sort_values('total-score', ascending=False).reset_index(drop=True)
        ctx['grades'] = grades

class determine_curve_participants(Stage):

    """
    Determine the subset of students that make up the curve.
    """

    def process(self, ctx):
        grades = ctx['grades']
        in_curve = np.repeat(True, grades.shape[0])
        grades['in-curve'] = in_curve

        if 'remove_incompletes' in self.params and self.params['remove_incompletes']:

            incompletes = set()

            def find_exceptions(row):
                sid = row['sid']
                if row['type'] == 'set_grade' and row['grade'] == 'I':
                    incompletes.add(sid)
            
            ctx['student_exceptions'].apply(find_exceptions, axis=1)

            def remove_incompletes(row):
                d = dict()
                if row['sid'] in incompletes:
                    d['in-curve'] = False
                return pd.Series(d)
            
            changes = grades.apply(remove_incompletes, axis=1)
            grades.update(changes)
        
        ctx['grades'] = grades

class create_buckets(Stage):

    """
    Create buckets table according to the given guidelines.
    """

    def process(self, ctx):
        targets = self.arg('targets')
        students = ctx['grades']

        def discretize_targets(no_students, targets):
            if sum(targets) != 1.0:
                print('WARNING: targets don\'t sum to 1.0')
            a = 0.0
            cumulative_targets = []
            for t in targets:
                a += t
                size_bucket = int(np.round(no_students * a))
                cumulative_targets.append(size_bucket)
            cumulative_targets[-1] = no_students
            return cumulative_targets
        
        discrete = discretize_targets(students.shape[0], targets)
        buckets = [dict(size=t) for t in discrete]
        ctx['buckets'] = pd.DataFrame(buckets)

class adjust_buckets(Stage):

    """
    Scan above and below to find more natural bucket locations to separate students.
    """

    def process(self, ctx):
        scan_frac, max_scan = self.args('scan_frac', 'max_scan')

        all_scores = ctx['grades']['total-score']
        buckets = ctx['buckets']['size']

        def adjust_buckets(scores, buckets, scan_frac=0.2, max_scan=10):
            new_buckets = []
            for i, b in enumerate(buckets):
                # adjust all but the last bucket - it has to include everyone anyway
                if i != len(buckets) - 1:
                    bucket_size = buckets[0] if i == 0 else buckets[i] - buckets[i-1]
                    num_scan = min(max_scan, int(bucket_size * scan_frac))
                    # scan width [b - num_scan, b + num_scan]
                    samples = np.array([])
                    bs = []
                    for new_b in range(b - num_scan, b + num_scan + 1):
                        score_diff = scores[new_b - 1] - scores[new_b]
                        bs.append((len(bs), new_b))
                        samples = np.append(samples, score_diff)
                    print('mean difference of', len(bs), 'differences around bucket: ', np.mean(samples))
                    max_i = max(bs, key=lambda x: samples[x[0]])[0]
                    print('maximum', bs[max_i][1], 'with diff', samples[max_i],
                        '({:2.2f}x)'.format(samples[max_i]/np.mean(samples)))
                    new_b = bs[max_i][1]
                    new_buckets.append(new_b)
                else:
                    new_buckets.append(b)
            return new_buckets

        new_buckets = adjust_buckets(all_scores, buckets, scan_frac, max_scan)
        ctx['buckets']['size'] = new_buckets

class create_boundaries(Stage):

    """
    Create the grading boundary thresholds from the buckets.
    The threshold is determined by the minimum score among the students in each bucket.
    """

    def process(self, ctx):
        buckets = ctx['buckets']
        all_scores = ctx['grades']['total-score']
        boundaries = [all_scores[s-1] for s in buckets['size']]
        buckets['boundary'] = boundaries
        ctx['buckets'] = buckets

class assign_letters(Stage):

    def process(self, ctx):
        final_floor, score_floor, letter_grades, gpas = self.args('final_floor', 'score_floor', 'grades', 'gpas')
        grades = ctx['grades']
        boundaries = ctx['buckets']['boundary']

        def compute_letter(row):
            score = row['total-score']
            final = row['final-score']

            if score < score_floor and (np.isnan(final) or final < final_floor):
                return pd.Series({'letter': 'F', 'gpa': 0})

            bucket_i = min(k for k in range(len(boundaries)) if boundaries[k] <= score)
            return pd.Series({
                'letter': letter_grades[bucket_i],
                'gpa': gpas[bucket_i]
            })

        letters = grades.apply(compute_letter, axis=1)
        grades = pd.concat([grades, letters], axis=1)

        print('avg gpa: {}'.format(grades['gpa'].mean()))
        ctx['grades'] = grades

class render_reports(Stage):

    def process(self, ctx):
        pass
