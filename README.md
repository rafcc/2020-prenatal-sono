# 2020-prenatal-sono
This project provides supplementary code for academic paper.

##  Requirements
This program has been verified under the following conditions.
### 3rd party program:
To execute this program, the darknet library which can be obtained from  https://github.com/pjreddie/darknet is required. How to use is written in https://pjreddie.com/darknet/.

### Library requirements:
- python2.7 or 3.6
- scikit-learn 0.19.1
- opencv-python                      3.4.0.12
- numpy                              1.14.1
- chainercv                          0.7.0
- lxml                               3.7.3
- gcc version 4.8.5

### Environment:
- CentOS Linux release 7.2.1511
- Driver Version: 418.39      
- CUDA Version: 10.1

### Computer configuration:
- Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz
- GeForce GTX 1080

## training

Prepare your own data as PASCAL-VOC format, and put the configuration file in the darknet configuration directory.
```
mv cfg_angledv2_20190729 darknet_root/cfg
```
Move to the darknet directory and execute learning.
```
cd darknet_root
./darknet detector train cfg/obj.data cfg/yolob2-obj.cfg
```
## trial.1
Modify ```trial.1/anomaly_angled_write_single20190803.py``` lines 9 to 14 of the AAA to get the heart part information and do the following:
```
python trial.1/anomaly_angled_write_single20190803.py
```
For multiple data, change lines 9 to 14 of ```trial.1/write_xml_for_mAP.py``` and do the following:
```
python trial.1/write_xml_for_mAP.py
```
To assess performance, modify ``` trial.1/mAP_test.py``` lines 6 to 7 and do the following:
```
python trial.1/mAP_test.py
```
## trial.2
To calculate the anomaly score, modify ``trial.2/`roc_scores_h.py ``` and ```trial.2/roc_scores_v.py ```lines 10 to 16  ,and do the following:

```
python trial.2/roc_socres_h.py
python trial.2/roc_scores_v.py
```
To evaluate the roc curve, modify ```trial.2/roc.py``` lines 6 to 7 and do the following:
```
python roc.py
```
The examples of the calculation result is stored in ```analysis/```.
