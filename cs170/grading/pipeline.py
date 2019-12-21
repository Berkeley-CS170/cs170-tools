import pandas as pd

class Stage:
    """
    One stage in a grade-processing pipeline.

    Has parameters that can be serialized, as well as optional defaults.
    """

    def __init__(self, **kwargs):
        if hasattr(self, '_defaults'):
            kwargs = {**self._defaults, **kwargs}
        if 'label' in kwargs:
            self._label = kwargs['label']
        self.params = kwargs

    def __str__(self):
        label = '<{}> '.format(self._label) if hasattr(self, '_label') else ''
        return '{}{}({})'.format(label, self.__class__.__name__, str(self.params))

    def arg(self, arg):
        if arg not in self.params:
            raise KeyError('{} not supplied for {}'.format(arg, self))
        return self.params[arg]

    def arg_opt(self, arg, default=None):
        if arg not in self.params: return default
        else: return self.params[arg]

    def args(self, *args):
        return [self.arg(arg) for arg in args]
    
    def process(self, ctx):
        raise NotImplementedError()

def freeze():
    pass

def mark():
    pass

class Context:
    """
    This context is passed between stages and pipelines, representing the current grading state.
    Any operations act on the latest state in the history, and refresh() adds a new state to the history.
    """

    def __init__(self, data={}, inputs={}):
        self.history = [(data, inputs)]

    def refresh(self, i=-1):
        """
        Make a copy of the state from iteration i and use it as the current state.
        By default, make a copy of the most recent state.
        """
        data, inputs = self.history[i]
        data_new = {}
        inputs_new = inputs.copy()

        for k in data:
            # create a deep copy of every data frame in data.
            # this may be inefficient, but it means good serialization
            data_new[k] = data[k].copy(deep=True)

        self.history.append((data_new, inputs_new))

    def refresh_with(self, data, inputs):
        self.history.append(data, inputs)

    def cur(self):
        return len(self.history) - 1
    
    def past(self, i):
        return self.history[i]

    def __getitem__(self, key):
        return self.history[-1][0][key]

    def __setitem__(self, key, value):
        self.history[-1][0][key] = value

    def __contains__(self, key):
        return key in self.history[-1][0]

    def get_or_create(self, key):
        if key not in self:
            self[key] = pd.DataFrame()
        return self[key]

    def inputs(self):
        return self.history[-1][1]

class Pipeline:
    
    def run(self, ctx):
        raise NotImplementedError()

class StagedPipeline(Pipeline):

    def __init__(self, stages=()):
        self.stages = stages

    def run(self, ctx):
        for stage in self.stages:
            print('running stage {}'.format(stage))
            stage.process(ctx)
            ctx.refresh()

class ConcatPipeline(Pipeline):

    def __init__(self, pipelines):
        self.pipelines = pipelines

    def run(self, ctx):
        for pipeline in self.pipelines:
            pipeline.run(ctx)
    
class MergePipeline(Pipeline):

    def __init__(self, pipelines):
        self.pipelines = pipelines

    def run(self, ctx):
        c = ctx.cur()

        past_is = {}
        for name in self.pipelines:
            ctx.refresh(c)
            self.pipelines[name].run(ctx)
            past_is[name] = ctx.cur()
        ctx.refresh(c)

        data_new = {}
        inputs_new = {}
        
        for name in past_is:
            data, inputs = ctx.past(past_is[name])
            for k in data:
                data_new[k + '_' + name] = data[k].copy(deep=True)

        ctx.refresh_with((data_new, inputs_new))

def pipeline(stages):
    return StagedPipeline(stages)

def concat_pipelines(pipelines):
    return ConcatPipeline(pipelines)

def merge_pipelines(pipelines):
    return MergePipeline(pipelines)

