# Kinetic Monte Carlo
The English Wiki page is a pretty good place to learn most of the important intro stuff for KMC
## Usage
There are tests which can be used as examples and *_skeleton files which you can use as a starting point
## Development
This is a passion project of mine and I just need more reasons to continue developing it. If you have any suggestions tell me and I'll gladly implement them. Hopefully one day it'll have a "release" patch 1.0

### numba
If you don't like / don't want numba just remove these
```bash
from numba import njit
@njit(nogil=True)
```
