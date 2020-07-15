******************
Benchmark Datasets
******************

`superres-tomo` is assoicated with a range of benchmark datasets. These can be used to benchmark
our implementations, to compare new methods, or simply as your own test.

+-----------------------+----------------+----------+-------------+--------------+---------+----------------------+
| Link                  | Real/Synthetic | Source   | Experiment  | Dimensions   | Number  | Notes                |
|                       |                |          |             |              | of data |                      |
+=======================+================+==========+=============+==============+=========+======================+
| http://tiny.cc/3hydsz | Synthetic      | DIV2K    | n/a         | 64x64        | 180,000 | Image/sinogram       |
|                       |                |          |             |              |         | pairs                |
+-----------------------+----------------+----------+-------------+--------------+---------+----------------------+
| http://tiny.cc/58zdsz | Real           | Micro CT | Reactor bed | 185x185x1001 | 1000    | 3rd dim is the       |
|                       |                |          |             |              |         | z-position           |
+-----------------------+----------------+----------+-------------+--------------+---------+----------------------+
| http://tiny.cc/ft0dsz | Real           | Micro CT | Reactor bed | 369x369x1001 | 1000    | Same experiment as   |
|                       |                |          |             |              |         | above; higher res    |
+-----------------------+----------------+----------+-------------+--------------+---------+----------------------+
| http://tiny.cc/0a0dsz | Real           | XRD-CT   | Reactor bed | 75x75x2038   |  1      | From same experiment |
|                       |                |          |             |              |         | as the data above.   |
|                       |                |          |             |              |         | 3rd dim is the       |
|                       |                |          |             |              |         | diffraction pattern. |
+-----------------------+----------------+----------+-------------+--------------+---------+----------------------+
| http://tiny.cc/jd0dsz | Real           | XRD-CT   | Reactor bed | 149x149x2038 | 1       | Same experiment as   |
|                       |                |          |             |              |         | above; higher-res    |
+-----------------------+----------------+----------+-------------+--------------+---------+----------------------+
