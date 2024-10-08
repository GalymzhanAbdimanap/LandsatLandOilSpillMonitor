## LandsatLandOilSpillMonitor


**Step 0.** 
```shell
pip install -r req.txt
```
**Step 1.** 
```shell
pip install "mmsegmentation>=1.0.0"
```
**Step 2.** Download weights from [drive](https://drive.google.com/file/d/1z8lNsyYRongWcH1BMVf4nPVDZMnU2HRE/view?usp=sharing) and put in config

**Step 3.**
```shell
uvicorn main:app --host 0.0.0.0 --port 5544
```
