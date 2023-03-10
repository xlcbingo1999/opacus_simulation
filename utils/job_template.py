class JobTemplate:
    def __init__(self, model_name, model, command, working_directory, 
                num_steps_arg, mem_request, target_dataset,
                 needs_data_dir=True, distributed=False):
        self._model_name = model_name
        self._model = model
        self._command = command
        self._working_directory = working_directory
        self._num_steps_arg = num_steps_arg
        self._mem_request = mem_request
        self._target_dataset = target_dataset
        self._needs_data_dir = needs_data_dir
        self._distributed = distributed

    @property
    def model_name(self):
        return self._model_name

    @property
    def model(self):
        return self._model

    @property
    def command(self):
        return self._command

    @property
    def working_directory(self):
        return self._working_directory

    @property
    def num_steps_arg(self):
        return self._num_steps_arg

    @property
    def mem_request(self):
        return self._mem_request

    @property
    def needs_data_dir(self):
        return self._needs_data_dir

    @property
    def distributed(self):
        return self._distributed

    @property
    def target_dataset(self):
        return self._target_dataset