# Environment for DexMV: Learning from Human Video

## Install

To use the environment implemented in this repo, install it in development mode (link locally):

```bash
python setup.py develop
# if use the system python: 
# python setup.py develop --user 
```

## File Structure

`hand_imitation`: the main lib to be installed: environment, model and kinematics retargeting
`examples`: examples file of usage of the code in this repository
`test_resources`: some data files to be used in some Unit Test (not enforced)
`docker`: docker file to configure the dependencies of this repo
