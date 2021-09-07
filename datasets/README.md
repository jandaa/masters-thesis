## Downloading Datasets

### Scannet

To get access to the scannet dataset, first you will have to complete the [ScanNet terms of Use](http://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf) and send the completed form to scannet@googlegroups.com.

Once you get a reply, they will provide you with a download script that you can place in the ``datasets/scannetv2/`` folder where a convenience script will help automate the download process. To download first make sure you have python2 installed. Then run the following:

```shell
cd datasets/scannetv2/
python download-instance-segmentation.py
```

**NOTE:** The full dataset size is ~1.3TB. Make sure you have enough space on your machine. 

### S3DIS

Getting access to S3DIS is slightly easier than ScanNet. Go to the [S3DIS](http://buildingparser.stanford.edu/dataset.html) website and scroll down to the *Download* section and click on the link under the *S3DIS Dataset* option. This should take you to fill out a form and then give you download options. Select the (I forget which one to download)