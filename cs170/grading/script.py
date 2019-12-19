import argparse
import sys
import os
import os.path
from .pipeline import Context

def find_grading_root(path, prev=None):
    """
    Recursively search the current working directory and its parents for a .gradingroot
    Stop when reaching a fixed point.
    """
    head, tail = os.path.split(path)
    grading_json = os.path.join(path, '.gradingroot')
    if os.path.exists(grading_json):
        return path
    elif (head, tail) != prev:
        return find_grading_root(head, (head, tail))
    else:
        return None

def main(name=None, pipelines=None):
    """
    Runs the grading file as a script.

    pipelines: dict where the keys are pipeline names, and the values are pipeline objects.
    """
    extra = ' for script {}'.format(name) if name else ''
    parser = argparse.ArgumentParser(
        description='Run grading{}'.format(extra)
    )

    subparsers = parser.add_subparsers(
        title='subcommands',
        description='valid subcommands',
        dest='command'
    )
    grade_command = subparsers.add_parser('grade',
        description='execute the grading script'
    )
    grade_command.add_argument('--pipeline', '-p', '-pipe')

    rerun_command = subparsers.add_parser('rerun',
        description='rerun a grading transcript'
    )
    rerun_command.add_argument('--pipeline', '-p', '-pipe', help='the pipeline to run on the transcript')
    rerun_command.add_argument('transcript', help='the grading trancript to rerun')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    root = find_grading_root(os.getcwd())

    if not root:
        print('Grading root not found.')
        print('Please create a .gradingroot file in the same directory as your grading scripts.')
        sys.exit(1)
    
    if args.command == 'grade':
        grade(name, pipelines, args)
    elif args.command == 'rerun':
        rerun(name, pipelines, args)

def select_pipeline(pipelines, pipeline):
    if not pipeline:
        if 'default' in pipelines:
            return pipelines['default']
        else:
            raise ValueError('No pipeline supplied, and no \'default\' pipeline found')
    else:
        if pipeline in pipelines:
            return pipelines[pipeline]
        else:
            raise ValueError('Pipeline \'{}\' not found among {}'.format(pipeline, ', '.join(pipelines.keys())))

def grade(name, pipelines, args):
    p = select_pipeline(pipelines, args.pipeline)
    p.run(Context())

def rerun(name, pipelines, args):
    p = select_pipeline(pipelines, args.pipeline)
    return None
