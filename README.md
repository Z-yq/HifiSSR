# Zero-Shot VC

环境安装

```shell
pip install -r requirments.txt

```

模型下载

```shell
modelscope download --model 'ACoderPassBy/HifiSSR' --local_dir './static'

```

快速使用

```shell
python ssr_infer.py --source_wav ./static/syz.wav --ref_wav ./static/p228_002.wav --out_wav ./ssred.wav --ckpt_path ./static
```


