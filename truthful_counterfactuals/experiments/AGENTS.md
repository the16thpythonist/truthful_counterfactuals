## AutoSlurm Computational Experiments

This project uses the [pycomex](https://github.com/the16thpythonist/pycomex) micro-framework for implementing and executing computational experiments. Each computational experiment is defined in a separate file. The files themselves can be executed to start the corresponding experiment. Each file accepts command line options related to the experiment parameters defined in the file. These command line options will override the default values defined in the file.

The execution / orchestration of these experiments is done using the [AutoSlurm](https://github.com/aimat-lab/AutoSlurm) package, which provides a convenient command line interface for scheduling SLURM jobs.

```bash
aslurm -cn euler cmd python experiment.py --PARAM1 100
```

**Note.** Use the config `-cn euler` by default.

## Experiment Sweeps

In combination, the `pycomex` and `AutoSlurm` packages allow for easy experiment sweeps. These sweeps are defined as bash files which themselves iterate over a set of parameters and schedules the corresponding experiments using the `aslurm` command as it is shown in the example below.

```bash
#!/bin/bash
# we need some kind of ID to identify the experiments later on for analysis
id='ex04_b'

params=(100 200 300)

for param in "${params[@]}"; do
    aslurm -cn euler cmd python experiment.py --PARAM1="$param" --__PREFIX__='${id}'
done
```