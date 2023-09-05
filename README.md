# Exploiting Aliasing for Manga Restoration
### [CVPR Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Xie_Exploiting_Aliasing_for_Manga_Restoration_CVPR_2021_paper.html) | [Arxiv](https://arxiv.org/abs/2105.06830) | [Project Website](http://www.cse.cuhk.edu.hk/~ttwong/papers/mangarestore/mangarestore.html) | [BibTex](#citation) 

## NOTE
This is a modified version made by me, bycloud. Please refer the original author [here](https://github.com/msxie92/MangaRestoration) for credit.
<!-- ------------------------------------------------------------------------------ -->
## Example Results 
Belows shows an example of our restored manga image. The image comes from the [Manga109 dataset](http://www.manga109.org/en/).

![Degraded](examples/Akuhamu_020.jpg)
![Restored](examples/Akuhamu_020_SR.png)
<!-- -------------------------------------------------------- -->

# Setup
Follow this YouTube [tutorial](https://youtu.be/uCTa4NUSwBs) or if you have any questions feel free to join my [discord](https://discord.gg/sE8R7e45MV) and ask there.

<!-- -------------------------------------------------------- -->
## Start
Clone this repository and place it anywhere you want on your PC.

<!-- ------------------------------------------------------------------- -->
## Pretrained models
Download the models below and create a folder called `release_model/` and put it under there.

[MangaRestoration](https://drive.google.com/file/d/1sazt7jlvfR6KEjOp9Tq2GpjMe04uRgtn/view?usp=sharing) 

<!-- -------------------------------------------------------- -->
## Setup environment
We are going to use Anaconda3, download [Anaconda3](https://www.anaconda.com/products/individual) if you don't have it.  

1. Create conda environment:
```
conda create -n EAMR python=3.8
conda activate EAMR
```
```
2. Install the dependencies
```
cd WHERE_YOU_CLONED_THIS_REPO
pip install -r requirements.txt
```
- *To reuse the created conda environment after you close the prompt, you just need to*:
```
conda activate EAMR
```
<!-- -------------------------------------------------------- -->
## Testing
 1. Create folder `datazip/manga1/test/` and `flist/manga1/`
 2. Place your test images under `datazip/manga1/test/`
 3. Run:
```
python scripts/flist.py --path datazip/manga1/test --output flist/manga1/test.flist
```
This generates a `test.flist` for your test images

 4. Run: GPU inference
```
python testreal.py -c configs/manga.json -n resattencv -s 256 -d gpu
```
4.1 Run: CPU inference
```
python testreal.py -c configs/manga.json -n resattencv -s 256 -d cpu
```
and your results will be under `MangaRestoration\release_model\resattencv_manga_cons256\results_real_00400\`

<!-- ------------------------------------------------------------------------------ -->
## Introduction 
As a popular entertainment art form, manga enriches the line drawings details with bitonal screentones. However, manga resources over the Internet usually show screentone artifacts because of inappropriate scanning/rescaling resolution. In this paper, we propose an innovative two-stage method to restore quality bitonal manga from degraded ones. Our key observation is that the aliasing induced by downsampling bitonal screentones can be utilized as informative clues to infer the original resolution and screentones. First, we predict the target resolution from the degraded manga via the Scale Estimation Network (SE-Net) with spatial voting scheme. Then, at the target resolution, we restore the region-wise bitonal screentones via the Manga Restoration Network (MR-Net) discriminatively, depending on the degradation degree. Specifically, the original screentones are directly restored in pattern-identifiable regions, and visually plausible screentones are synthesized in pattern-agnostic regions. Quantitative evaluation on synthetic data and visual assessment on real-world cases illustrate the effectiveness of our method.




## Copyright and License
You are granted with the [LICENSE](./LICENSE) for both academic and commercial usages.

<!-- ------------------------------------------------------------------- -->
## Citation
If any part of our paper and code is helpful to your work, please generously cite with:
```
@inproceedings{xie2021exploiting,
  author = {Minshan Xie and Menghan Xia and Tien-Tsin Wong},
  title = {Exploiting Aliasing for Manga Restoration},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition},
  year = {2021},
  pages = {13405--13414}
}
```

## Reference
- [PEN-Net](https://github.com/researchmm/PEN-Net-for-Inpainting)
- [ResidualAttentionNetwork](https://github.com/tengshaofeng/ResidualAttentionNetwork-pytorch)
